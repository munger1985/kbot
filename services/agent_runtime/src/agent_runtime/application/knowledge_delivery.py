"""Runtime 的有界真实载荷交付；检索主命中与模型上下文选择分别保留。"""

from dataclasses import dataclass, replace
import asyncio
from hashlib import sha256
import re

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.runtime import (
    CitationLabel, CitationLabelMap, DeliveredEvidence, GroundedAnswerArtifact, ModelEvidenceDelivery,
)
from knowledge_algorithms.retrieval.selection import EvidenceGroup, PayloadCost, PayloadUnit, select_evidence_groups


@dataclass(frozen=True, slots=True)
class MaterializedEvidence:
    item: object
    data: bytes
    cost: PayloadCost
    input_kind: str


@dataclass(frozen=True, slots=True)
class ModelEvidenceReply:
    body: str
    used_labels: tuple[str, ...]
    sent_at: int


class EvidenceDeliveryError(RuntimeError):
    def __init__(self, state="UNKNOWN_DELIVERY"):
        require(state in {"FAILED", "UNKNOWN_DELIVERY"}, "模型发送失败状态无效")
        super().__init__("模型证据交付失败")
        self.state = state


@dataclass(frozen=True, slots=True)
class CompositionResult:
    answer: GroundedAnswerArtifact | None
    delivery: ModelEvidenceDelivery
    labels: CitationLabelMap
    excluded: tuple[tuple[object, str], ...]
    selection: object


class KnowledgeEvidenceComposer:
    def __init__(self, *, tokenizer, image_cost, model_gateway, artifact_recorder,
                 max_payload_bytes, max_materialized_items, max_cover_states,
                 image_mime_types=frozenset({"image/png", "image/jpeg", "image/webp"})):
        require(all(type(n) is int and n > 0 for n in
                    (max_payload_bytes, max_materialized_items, max_cover_states)), "模型交付预算无效")
        self.tokenizer, self.image_cost, self.gateway, self.recorder = tokenizer, image_cost, model_gateway, artifact_recorder
        self.max_payload_bytes, self.max_items, self.max_cover_states = max_payload_bytes, max_materialized_items, max_cover_states
        self.image_mime_types = image_mime_types

    async def compose(self, *, run_ref, turn_ref, model_snapshot_ref, evidence_set, content_port,
                      question, budget, search_only=False):
        require(evidence_set.owner_binding_ref == content_port.owner, "内容端口与检索 owner 不一致")
        if search_only:
            delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "NOT_APPLICABLE", (), None)
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, CitationLabelMap(run_ref, ()), (), evidence_set.context_selection)
        require(type(question) is str and question.strip(), "问答缺少原始问题")
        items = {item.key: item for item in evidence_set.evidence}
        primary = {hit.evidence_key for hit in evidence_set.primary_hits}
        # 先读原有选择，再有界尝试其他组，避免单个大对象失败清空所有有效结果。
        ordered = tuple(dict.fromkeys(evidence_set.context_selection.selected_evidence_keys
                       + tuple(items)))
        materialized, excluded, consumed = {}, [], 0
        for position, key in enumerate(ordered):
            item = items[key]
            if position >= self.max_items or consumed + item.byte_count > self.max_payload_bytes:
                excluded.append((key, "MATERIALIZATION_BUDGET"))
                continue
            try:
                data = bytearray()
                stream = content_port.content(key, max_bytes=min(item.byte_count, self.max_payload_bytes - consumed))
                try:
                    async for block in stream:
                        require(type(block) is bytes and 0 < len(block) <= 65536, "内容流块格式错误")
                        consumed += len(block)
                        require(consumed <= self.max_payload_bytes and len(data) + len(block) <= item.byte_count,
                                "内容流超过冻结预算")
                        data.extend(block)
                finally:
                    await stream.aclose()
                raw = bytes(data)
                require(len(raw) == item.byte_count and sha256(raw).hexdigest() == item.content_hash,
                        "交付对象内容摘要或字节数错误")
                if item.mime_type.startswith("image/"):
                    require(item.mime_type in self.image_mime_types, "当前模型未登记该图片格式")
                    cost = PayloadCost(self.image_cost(raw, item.mime_type), 1, len(raw))
                    kind = "IMAGE"
                else:
                    require(item.mime_type == "text/plain", "模型交付只允许明确的文本或图片")
                    cost = PayloadCost(self.tokenizer.count(raw.decode("utf-8")), 0, len(raw))
                    kind = "TEXT"
                materialized[key] = MaterializedEvidence(item, raw, cost, kind)
            except Exception:
                # 不输出原文、路径、异常绑定参数；也不把失败伪装为没有召回。
                excluded.append((key, "MATERIALIZATION_FAILED"))
        validity = await content_port.revalidate(tuple(materialized))
        require(set(validity) <= set(materialized), "内容复核返回未请求的身份")
        for key in tuple(materialized):
            if key not in validity:
                excluded.append((key, "ACCESS_REVOKED"))
                materialized.pop(key)
        def choose():
            groups = []
            for manifest in evidence_set.groups:
                if manifest.anchor_key not in materialized:
                    continue
                keys = tuple(dict.fromkeys((manifest.anchor_key,) + manifest.linked_keys))
                members = tuple(PayloadUnit(key, materialized[key].cost) for key in keys if key in materialized)
                groups.append(EvidenceGroup(manifest.anchor_key.stable_key, manifest.anchor_key, members,
                                            tuple(edge.relation_key for edge in manifest.relations)))
            selection = select_evidence_groups(tuple(groups), targets=evidence_set.plan.targets,
matches=tuple(match for match in evidence_set.matches if match.evidence_key in
                    {member.key for group in groups for member in group.members}),
                budget=budget, max_cover_states=self.max_cover_states)
            selected = selection.selected_evidence_keys
            labels = CitationLabelMap(run_ref, tuple(CitationLabel(run_ref, "D" + str(index), key,
                "PRIMARY" if key in primary else "REFERENCED", items[key].locators)
                for index, key in enumerate(selected, 1)))
            prepared = tuple(DeliveredEvidence(key, label.label, items[key].content_hash,
                materialized[key].input_kind, materialized[key].cost) for key, label in zip(selected, labels.entries))
            return selection, selected, labels, prepared

        # 两次有界重选；单个组撤权不清空其他仍合法、可交付的证据。
        for attempt in range(3):
            selection, selected, labels, prepared = choose()
            if not prepared:
                delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "FAILED", (), None)
                await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
                return CompositionResult(None, delivery, labels, tuple(excluded), selection)
            valid = set(await content_port.revalidate(selected))
            require(valid <= set(selected), "发送前复核伪造证据身份")
            if valid == set(selected):
                break
            for key in set(selected) - valid:
                materialized.pop(key, None)
                excluded.append((key, "ACCESS_REVOKED_BEFORE_SEND"))
            if attempt == 2:
                delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "FAILED", (), None)
                await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
                raise AlgorithmError(ErrorCode.SCOPE_CHANGED, "授权持续改变，已停止发送", stage="delivery")
        delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "PREPARED", prepared, None)
        await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
        parts = tuple({"label": label.label, "kind": materialized[label.evidence_key].input_kind,
            "mime_type": items[label.evidence_key].mime_type, "data": materialized[label.evidence_key].data}
            for label in labels.entries)
        try:
            reply = await self.gateway.generate(question=question, evidence_parts=parts,
                                                 model_snapshot_ref=model_snapshot_ref)
        except EvidenceDeliveryError as exc:
            delivery = replace(delivery, status=exc.state)
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, labels, tuple(excluded), selection)
        except Exception:
            delivery = replace(delivery, status="UNKNOWN_DELIVERY")
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, labels, tuple(excluded), selection)
        except asyncio.CancelledError:
            # 超时/取消不证明请求没有发送；审计不得停留在 PREPARED。
            delivery = replace(delivery, status="UNKNOWN_DELIVERY")
            await asyncio.shield(self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery))
            raise
        if (not isinstance(reply, ModelEvidenceReply) or type(reply.sent_at) is not int or reply.sent_at <= 0
                or type(reply.body) is not str or type(reply.used_labels) is not tuple):
            delivery = replace(delivery, status="UNKNOWN_DELIVERY")
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, labels, tuple(excluded), selection)
        delivery = replace(delivery, status="SENT", sent_at=reply.sent_at)
        await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
        allowed = {label.label: label for label in labels.entries}
        actual = tuple(dict.fromkeys(re.findall(r"\[([A-Z]+[0-9]+)\]", reply.body)))
        require(reply.used_labels == actual and set(actual) <= set(allowed),
                "回答引用了未发送、伪造或顺序不一致的证据")
        require(bool(actual), "有证据的正式回答必须明确引用实际发送来源")
        incomplete = bool(excluded) or evidence_set.status != "COMPLETE" or any(
            status.status not in {"MATCHED", "SATISFIED"} for status in selection.coverage)
        answer = GroundedAnswerArtifact(run_ref, reply.body, actual, tuple(allowed[label] for label in actual),
                                         "PARTIAL" if incomplete else "COMPLETE")
        await self.recorder.record(run_ref, turn_ref, "GROUNDED_ANSWER", answer)
        return CompositionResult(answer, delivery, labels, tuple(excluded), selection)
