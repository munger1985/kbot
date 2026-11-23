"""Runtime 的有界真实载荷交付；检索主命中与模型上下文选择分别保留。"""

from dataclasses import dataclass, replace
import asyncio
from hashlib import sha256
import re
import time

from loguru import logger
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.runtime import (
    CitationLabel, CitationLabelMap, DeliveredEvidence, GroundedAnswerArtifact, ModelEvidenceDelivery,
)
from knowledge_algorithms.retrieval.selection import EvidenceGroup, PayloadCost, PayloadUnit, select_evidence_groups


EVIDENCE_READ_CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class MaterializedEvidence:
    item: object
    data: bytes
    cost: PayloadCost
    input_kind: str


@dataclass(frozen=True, slots=True)
class QuestionImage:
    """本轮会话附件，不属于知识库证据，也没有知识库引用标签。"""

    mime_type: str
    data: bytes
    ocr_text: str = ""
    ocr_truncated: bool = False


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
                 image_mime_types=frozenset({"image/png", "image/jpeg", "image/webp"}), estimate_tokens=None):
        require(all(type(n) is int and n > 0 for n in
                    (max_payload_bytes, max_materialized_items, max_cover_states)), "模型交付预算无效")
        self.tokenizer, self.image_cost, self.gateway, self.recorder = tokenizer, image_cost, model_gateway, artifact_recorder
        self.max_payload_bytes, self.max_items, self.max_cover_states = max_payload_bytes, max_materialized_items, max_cover_states
        self.image_mime_types = image_mime_types
        self.estimate_tokens = estimate_tokens or (lambda item: item.estimated_cost.tokens)

    async def _materialize(self, port, item):
        data = bytearray()
        try:
            stream = port.content(item.key, max_bytes=item.byte_count)
            try:
                async for block in stream:
                    require(type(block) is bytes and 0 < len(block) <= 65536, "内容流块格式错误")
                    require(len(data) + len(block) <= item.byte_count, "内容流超过冻结预算")
                    data.extend(block)
            finally:
                await stream.aclose()
            raw = bytes(data)
            require(len(raw) == item.byte_count and sha256(raw).hexdigest() == item.content_hash,
                    "交付对象内容摘要或字节数错误")
            image = item.mime_type in self.image_mime_types
            tokens = self.image_cost(raw, item.mime_type) if image else self.tokenizer.count(raw.decode("utf-8"))
            return (MaterializedEvidence(item, raw, PayloadCost(tokens, int(image), len(raw)),
                                         "IMAGE" if image else "TEXT"), len(raw))
        except Exception:
            # 失败不释放已经读取的字节预算，也不记录未经筛选的异常内容。
            return None, len(data)

    async def compose(self, *, run_ref, turn_ref, model_snapshot_ref, evidence_set, content_port,
                      question, budget, search_only=False, question_images=(),
                      ocr_model_snapshot_ref=None, transcription_requested=False, on_preview=None):
        require(evidence_set.owner_binding_ref == content_port.owner, "内容端口与检索 owner 不一致")
        if search_only:
            delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "NOT_APPLICABLE", (), None)
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, CitationLabelMap(run_ref, ()), (), evidence_set.context_selection)
        require(type(question) is str and question.strip(), "问答缺少原始问题")
        image_count = len(question_images) or None
        if question_images:
            require(len(question_images) <= 4 and all(isinstance(image, QuestionImage)
                    for image in question_images), "提问图片格式无效", stage="question_image")
            complete_ocr = all(image.ocr_text and not image.ocr_truncated for image in question_images)
            if ocr_model_snapshot_ref and (not self.image_mime_types or (transcription_requested and complete_ocr)):
                return await self._ocr_answer(run_ref, turn_ref, ocr_model_snapshot_ref, evidence_set,
                                              question_images, budget, transcription_requested)
            if not self.image_mime_types:
                raise AlgorithmError(ErrorCode.IMAGE_INPUT_UNSUPPORTED, "当前 Agent 未配置 OCR 或视觉模型",
                                     stage="vision_model")
            require(all(image.mime_type in self.image_mime_types and type(image.data) is bytes and image.data
                    for image in question_images), "提问图片格式无效", stage="question_image")
            image_bytes = sum(len(image.data) for image in question_images)
            image_cost = PayloadCost(sum(self.image_cost(image.data, image.mime_type) for image in question_images),
                                     len(question_images), image_bytes)
            if not image_cost.fits(budget) or image_bytes > self.max_payload_bytes:
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "提问图片超出视觉模型交付预算",
                                     stage="question_image_budget")
            budget = PayloadCost(budget.tokens - image_cost.tokens, budget.images - image_cost.images,
                                 budget.byte_count - image_cost.byte_count)
            # 识别文字只是辅助信息，不能挤掉原图；只使用剩余预算中的一小部分。
            ocr_budget = min(4096, budget.tokens, budget.byte_count, self.max_payload_bytes - image_bytes)
            enriched, ocr_bytes = [], 0
            for image in question_images:
                raw = image.ocr_text.encode("utf-8")
                text = raw[:ocr_budget - ocr_bytes].decode("utf-8", errors="ignore")
                added = len(text.encode("utf-8"))
                ocr_bytes += added
                enriched.append(replace(image, ocr_text=text,
                                        ocr_truncated=image.ocr_truncated or added < len(raw)))
            question_images = tuple(enriched)
            budget = PayloadCost(budget.tokens - ocr_bytes, budget.images, budget.byte_count - ocr_bytes)
            image_bytes += ocr_bytes
        else:
            image_bytes = 0
        started = time.monotonic()
        materialization_limit = self.max_payload_bytes - image_bytes
        items = {item.key: item for item in evidence_set.evidence}
        primary = {hit.evidence_key for hit in evidence_set.primary_hits}
        # 先用冻结元数据选组；从不为未选中的候选请求正文。
        costs = {key: PayloadCost(self.estimate_tokens(item), int(item.mime_type in self.image_mime_types),
                                  item.byte_count)
                 for key, item in items.items() if 0 < item.byte_count <= materialization_limit
                 and (item.mime_type == "text/plain" or item.mime_type in self.image_mime_types)}
        materialized, excluded = {}, []
        reserved_bytes = read_count = 0

        def choose(available):
            groups = []
            for manifest in evidence_set.groups:
                if manifest.anchor_key not in available:
                    continue
                keys = tuple(dict.fromkeys((manifest.anchor_key,) + manifest.linked_keys))
                members = tuple(PayloadUnit(key, available[key]) for key in keys if key in available)
                groups.append(EvidenceGroup(manifest.anchor_key.stable_key, manifest.anchor_key, members,
                                            tuple(edge.relation_key for edge in manifest.relations)))
            selectable = {member.key for group in groups for member in group.members}
            return select_evidence_groups(tuple(groups), targets=evidence_set.plan.targets,
                matches=tuple(match for match in evidence_set.matches if match.evidence_key in selectable),
                budget=budget, max_cover_states=self.max_cover_states)

        async def materialize_selection():
            nonlocal read_count, reserved_bytes
            while True:
                selection = choose(costs)
                pending = [key for key in selection.selected_evidence_keys if key not in materialized]
                if not pending:
                    return selection
                for start in range(0, len(pending), EVIDENCE_READ_CONCURRENCY):
                    batch = []
                    for key in pending[start:start + EVIDENCE_READ_CONCURRENCY]:
                        item = items[key]
                        if read_count >= self.max_items or reserved_bytes + item.byte_count > materialization_limit:
                            costs.pop(key)
                            excluded.append((key, "MATERIALIZATION_BUDGET"))
                            continue
                        read_count += 1
                        reserved_bytes += item.byte_count
                        batch.append(item)
                    if not batch:
                        continue
                    values = await asyncio.gather(*(self._materialize(content_port, item) for item in batch),
                                                  return_exceptions=True)
                    for item, result in zip(batch, values, strict=True):
                        if isinstance(result, BaseException):
                            raise result
                        value, received = result
                        reserved_bytes -= item.byte_count - received
                        if value is None:
                            costs.pop(item.key)
                            excluded.append((item.key, "MATERIALIZATION_FAILED"))
                        else:
                            materialized[item.key] = value
                            costs[item.key] = value.cost

        # 两次有界重选；单个组撤权不清空其他仍合法、可交付的证据。
        for attempt in range(3):
            selection = await materialize_selection()
            selected = selection.selected_evidence_keys
            if not selected:
                if question_images:
                    break
                delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "FAILED", (), None)
                await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
                return CompositionResult(None, delivery, CitationLabelMap(run_ref, ()), tuple(excluded), selection)
            # 读完先复核；发送前再检查，持续变更则按原规则停止交付。
            first = set(await content_port.revalidate(selected))
            require(first <= set(selected), "内容复核返回未请求的身份")
            valid = set(await content_port.revalidate(tuple(key for key in selected if key in first))) if first else set()
            require(valid <= first, "发送前复核伪造证据身份")
            if valid == set(selected):
                break
            for key in set(selected) - valid:
                materialized.pop(key, None)
                costs.pop(key, None)
                excluded.append((key, "ACCESS_REVOKED_BEFORE_SEND"))
            if attempt == 2:
                delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "FAILED", (), None,
                                                 question_image_count=image_count)
                await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
                raise AlgorithmError(ErrorCode.SCOPE_CHANGED, "授权持续改变，已停止发送", stage="delivery")
        labels = CitationLabelMap(run_ref, tuple(CitationLabel(run_ref, "D" + str(index), key,
            "PRIMARY" if key in primary else "REFERENCED", items[key].locators)
            for index, key in enumerate(selected, 1)))
        prepared = tuple(DeliveredEvidence(key, label.label, items[key].content_hash,
            materialized[key].input_kind, materialized[key].cost) for key, label in zip(selected, labels.entries))
        logger.info("问答证据准备完成 | event=agent.knowledge.prepared | run_id={} | candidates={} | reads={} | delivered={} | duration_ms={:.1f}",
                    run_ref, len(items), read_count, len(selected), (time.monotonic() - started) * 1000)
        delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "PREPARED", prepared, None,
                                         question_image_count=image_count)
        await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
        parts = tuple({"label": label.label, "kind": materialized[label.evidence_key].input_kind,
            "mime_type": items[label.evidence_key].mime_type, "data": materialized[label.evidence_key].data}
            for label in labels.entries)
        model_started = time.monotonic()
        try:
            model_input = {"question_images": question_images} if question_images else {}
            if on_preview is not None:
                model_input["on_preview"] = on_preview
            reply = await self.gateway.generate(question=question, evidence_parts=parts,
                                                 model_snapshot_ref=model_snapshot_ref, **model_input)
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
        finally:
            logger.info("问答模型调用结束 | event=agent.knowledge.model_finished | run_id={} | vision={} | duration_ms={:.1f}",
                        run_ref, bool(self.image_mime_types), (time.monotonic() - model_started) * 1000)
        if (not isinstance(reply, ModelEvidenceReply) or type(reply.sent_at) is not int or reply.sent_at <= 0
                or type(reply.body) is not str or not reply.body.strip() or type(reply.used_labels) is not tuple):
            delivery = replace(delivery, status="UNKNOWN_DELIVERY")
            await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
            return CompositionResult(None, delivery, labels, tuple(excluded), selection)
        delivery = replace(delivery, status="SENT", sent_at=reply.sent_at)
        await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
        allowed = {label.label: label for label in labels.entries}
        actual = tuple(dict.fromkeys(re.findall(r"\[([A-Z]+[0-9]+)\]", reply.body)))
        require(reply.used_labels == actual and set(actual) <= set(allowed),
                "回答引用了未发送、伪造或顺序不一致的证据")
        require(bool(actual or question_images), "有证据的正式回答必须明确引用实际发送来源")
        # 观察用户图片不依赖知识库命中；使用库内资料时仍保留原有覆盖判断。
        incomplete = (bool(actual) and (bool(excluded) or evidence_set.status != "COMPLETE" or any(
            status.status not in {"MATCHED", "SATISFIED"} for status in selection.coverage)))
        answer = GroundedAnswerArtifact(run_ref, reply.body, actual, tuple(allowed[label] for label in actual),
                                         "PARTIAL" if incomplete else ("UNKNOWN" if not actual else "COMPLETE"))
        await self.recorder.record(run_ref, turn_ref, "GROUNDED_ANSWER", answer)
        return CompositionResult(answer, delivery, labels, tuple(excluded), selection)

    async def _ocr_answer(self, run_ref, turn_ref, model_snapshot_ref, evidence_set, images, budget,
                          transcription_requested):
        readable = sum(bool(image.ocr_text) for image in images)
        if not readable:
            raise AlgorithmError(ErrorCode.NO_ACCESSIBLE_PAYLOAD, "未取得可读的图片文字识别结果",
                                 stage="question_ocr")
        heading = "以下是图片中的识别文字，可能有识别误差。"
        if not transcription_requested:
            heading = "当前 Agent 仅配置 OCR，以下是识别文字，未解读图形、布局或流程。"
        sections = [heading]
        for index, image in enumerate(images, 1):
            if not image.ocr_text:
                sections.append(f"**图片 {index}**：未取得可读文字。")
                continue
            # 原图可能含有形似引用的字符串。作为转写显示，不生成可点击的知识库引用。
            text = re.sub(r"\[([A-Z]+[0-9]+)\]", r"［\1］", image.ocr_text)
            label = f"**图片 {index}**" + ("（结果已截断）" if image.ocr_truncated else "")
            sections.append(label + "\n\n" + "\n".join("    " + line for line in text.splitlines()))
        selection = select_evidence_groups((), targets=evidence_set.plan.targets, matches=(), budget=budget,
                                           max_cover_states=self.max_cover_states)
        delivery = ModelEvidenceDelivery(run_ref, model_snapshot_ref, "SENT", (), int(time.time()),
                                         question_image_count=readable, question_image_kind="OCR")
        answer = GroundedAnswerArtifact(run_ref, "\n\n".join(sections), (), (), "UNKNOWN")
        await self.recorder.record(run_ref, turn_ref, "MODEL_EVIDENCE_DELIVERY", delivery)
        await self.recorder.record(run_ref, turn_ref, "GROUNDED_ANSWER", answer)
        return CompositionResult(answer, delivery, CitationLabelMap(run_ref, ()), (), selection)
