"""从既有 EvidencePlanner 提取的结构边界切分；不沿用字符数估算 token。"""

from contextlib import aclosing
from dataclasses import replace
from typing import AsyncIterator

from ..contracts.content import ContentKind
from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import AppBackendBinding, canonical_json, fingerprint
from ..contracts.wire import encode
from ..execution import Deadline
from .contracts import (AllowedContent, AllowedDocumentReader, ModelGateway, ModelInput, ProjectedUnit,
                        ProjectionBatch, ProjectionProfile, SearchUnit, Tokenizer)
from .splitting import REVISION as SPLIT_REVISION, plan_chunks


class ProjectionBuilder:
    def __init__(self, binding: AppBackendBinding, tokenizer: Tokenizer, model_gateway: ModelGateway | None):
        self.binding, self.tokenizer, self.model_gateway = binding, tokenizer, model_gateway

    def projection_ref(self, document: AllowedDocumentReader, profile: ProjectionProfile) -> str:
        self.binding.check_owner(document.binding.binding_ref)
        return fingerprint({"binding": self.binding.binding_ref,
            "document_version": document.document_version_ref, "parse_run": document.parse_run_ref,
            "view_ref": document.allowed_view_ref, "view_hash": document.allowed_view_hash,
            "profile": encode(profile), "tokenizer": self.tokenizer.revision, "split": SPLIT_REVISION})

    async def build(self, document: AllowedDocumentReader, profile: ProjectionProfile, *,
                    load_saved=None, max_batch_bytes: int | None = None,
                    projection_ref: str | None = None, reuse_representations=None) -> AsyncIterator[ProjectionBatch]:
        """按固定单元边界生成批次；持久批次必须与当前规划精确一致才可复用。"""
        planned_ref = self.projection_ref(document, profile)
        projection_ref = projection_ref or planned_ref
        sequence = 0
        async with aclosing(self._unit_batches(document, profile, projection_ref, max_batch_bytes)) as batches:
            async for pending, deadline in batches:
                sequence += 1
                saved = await load_saved(sequence, deadline) if load_saved is not None else None
                if saved is not None:
                    require(saved.projection_ref == projection_ref and saved.sequence == sequence
                            and tuple(item.unit for item in saved.units) == pending,
                            "已提交索引批次与当前检索单元规划不一致")
                    yield saved
                    continue
                projected = []
                remaining = max_batch_bytes
                for unit in pending:
                    await deadline.checkpoint(stage="projection")
                    prior = await reuse_representations(unit, deadline) if reuse_representations is not None else None
                    item = await self.represent_unit(unit, profile, deadline, prior)
                    remaining = consume_batch_bytes(item, remaining)
                    projected.append(item)
                units = tuple(projected)
                yield ProjectionBatch(projection_ref, sequence, units, fingerprint(encode(units)))

    async def plan_batch_count(self, document: AllowedDocumentReader, profile: ProjectionProfile, *,
                               max_batch_bytes: int | None = None) -> int:
        """同一路径预数批次，只保留单批，不调用表示模型。"""
        count = 0
        async with aclosing(self._unit_batches(document, profile, self.projection_ref(document, profile),
                                                max_batch_bytes)) as batches:
            async for _ in batches:
                count += 1
        return count

    async def _unit_batches(self, document, profile, projection_ref, max_batch_bytes):
        count, chars, pending = 0, 0, []
        seen = set()
        deadline = Deadline(profile.timeout_seconds)
        used_bytes = 0
        iterator = document.contents()

        try:
            while True:
                try:
                    content = await deadline.run(anext(iterator), stage="projection_read")
                except StopAsyncIteration:
                    break
                self.binding.check_owner(content.key.owner_binding_ref)
                require((content.key.document_version_ref, content.key.parse_run_ref)
                        == (document.document_version_ref, document.parse_run_ref), "允许视图版本或解析代次错误")
                require(content.key not in seen, "允许视图重复交付同一对象")
                seen.add(content.key)
                chars += len(content.text) + sum(map(len, content.heading_path))
                if chars > profile.max_input_chars or len(seen) > profile.max_units:
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "允许视图超过投影输入预算", stage="projection")
                if content.kind in {ContentKind.HEADER, ContentKind.FOOTER, ContentKind.DOCUMENT}:
                    continue
                chunks = self._chunks(content, profile)
                for start, end, context in chunks:
                    await deadline.checkpoint(stage="projection")
                    text = content.text[start:end]
                    sources = []
                    for source in content.sources:
                        a, b = max(start, source.unit_start), min(end, source.unit_end)
                        if a < b:
                            source_start = source.source_start + a - source.unit_start
                            sources.append(replace(source, unit_start=a - start, unit_end=b - start,
                                source_start=source_start, source_end=source_start + b - a))
                    unit_key = fingerprint({"source": content.key.stable_key, "span": (start, end),
                                            "projection": projection_ref})
                    projected_key = replace(content.key, projection_ref=projection_ref)
                    # 同一对象多个片段保留独立 unit_key，证据键也绑定确切片段映射。
                    projected_key = replace(projected_key, object_key=unit_key)
                    unit = SearchUnit(projected_key, content.key, unit_key, content.kind, text, context, tuple(sources),
                        content.locators, content.provenance, document.allowed_view_ref, document.allowed_view_hash,
                        content.content_handle if start == 0 else None)
                    count += 1
                    if count > profile.max_units:
                        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "检索单元超过预算", stage="projection")
                    size = planned_unit_bytes(unit, profile, max_batch_bytes)
                    if pending and (len(pending) >= profile.batch_size or
                                    max_batch_bytes is not None and used_bytes + size > max_batch_bytes):
                        yield tuple(pending), deadline
                        pending = []
                        used_bytes = 0
                        # 每批重新计时；磁盘与数据库提交不消耗下一批的模型时间。
                        deadline = Deadline(profile.timeout_seconds)
                    pending.append(unit)
                    used_bytes += size
            if pending:
                yield tuple(pending), deadline
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def represent_unit(self, unit, profile, deadline, prior: ProjectedUnit | None = None):
        """只允许明确的单图输入拒绝成为缺口；原图与其他通道不受影响。"""
        representations = []
        skipped = set()
        reusable = {item.space_id: item for item in prior.representations} if prior is not None else {}
        previous_skips = prior.skipped_image_spaces if prior is not None else None
        for spec in profile.representations:
            if not uses_representation(unit, spec):
                continue
            if spec.input_kind == "IMAGE" and previous_skips and spec.space_id in previous_skips:
                skipped.add(spec.space_id)
                continue
            representation = reusable.get(spec.space_id)
            if representation is None:
                if self.model_gateway is None:
                    raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "未注入所需表示模型", stage="projection")
                value = ModelInput("TEXT", text="\n".join(v for v in (unit.context, unit.text) if v),
                    input_role="DOCUMENT") if spec.input_kind == "TEXT" else ModelInput(
                        "IMAGE", image_handle=unit.content_handle, input_role="DOCUMENT")
                try:
                    representation = await deadline.run(self.model_gateway.represent(value, spec), stage="projection_model")
                except AlgorithmError as exc:
                    if value.kind != "IMAGE" or exc.code != ErrorCode.IMAGE_INPUT_UNSUPPORTED:
                        raise
                    skipped.add(spec.space_id)
                    continue
            representation.validate(spec, profile.max_representation_elements)
            representations.append(representation)
        return ProjectedUnit(unit, tuple(representations), frozenset(skipped) if skipped else None)

    def _chunks(self, content: AllowedContent, profile: ProjectionProfile):
        return plan_chunks(content, profile.max_tokens, self.tokenizer)


def uses_representation(unit, spec):
    return bool(unit.text or unit.context) if spec.input_kind == "TEXT" else unit.content_handle is not None


def planned_unit_bytes(unit, profile, limit):
    """单向量按维度预留保守编码量；形状不定的多向量独占一个批次。"""
    if limit is None:
        return 0
    size = len(canonical_json(encode(ProjectedUnit(unit, ()))).encode("utf-8")) + 1
    variable = False
    for spec in profile.representations:
        if not uses_representation(unit, spec):
            continue
        if spec.kind == "MULTIVECTOR":
            variable = True
        else:
            size += 128 + 32 * spec.dimension
    if size > limit:
        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "索引单元载荷超过单批硬上限", stage="projection_batch")
    return limit if variable else size


def consume_batch_bytes(item: ProjectedUnit, remaining: int | None) -> int | None:
    """逐单元核算编码体积，避免超限批次先在内存中完整堆积。"""
    if remaining is None:
        return None
    remaining -= len(canonical_json(encode(item)).encode("utf-8")) + 1
    if remaining < 0:
        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "索引批次超过载荷预算", stage="projection_batch")
    return remaining
