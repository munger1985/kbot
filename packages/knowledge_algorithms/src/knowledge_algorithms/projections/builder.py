"""从既有 EvidencePlanner 提取的结构边界切分；不沿用字符数估算 token。"""

from dataclasses import asdict, replace
import re
from typing import AsyncIterator

from ..contracts.content import ContentKind
from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import AppBackendBinding, fingerprint
from ..contracts.wire import encode
from ..execution import Deadline
from .contracts import (AllowedContent, AllowedDocumentReader, ModelGateway, ModelInput, ProjectedUnit,
                        ProjectionBatch, ProjectionProfile, SearchUnit, Tokenizer)


class ProjectionBuilder:
    def __init__(self, binding: AppBackendBinding, tokenizer: Tokenizer, model_gateway: ModelGateway | None):
        self.binding, self.tokenizer, self.model_gateway = binding, tokenizer, model_gateway

    async def build(self, document: AllowedDocumentReader, profile: ProjectionProfile) -> AsyncIterator[ProjectionBatch]:
        self.binding.check_owner(document.binding.binding_ref)
        profile_data = encode(profile)
        projection_ref = fingerprint({"binding": self.binding.binding_ref,
            "document_version": document.document_version_ref, "parse_run": document.parse_run_ref,
            "view_ref": document.allowed_view_ref, "view_hash": document.allowed_view_hash,
            "profile": profile_data, "tokenizer": self.tokenizer.revision})
        deadline = Deadline(profile.timeout_seconds)
        count, chars, sequence, pending = 0, 0, 0, []
        seen = set()
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
                for spans in chunks:
                    await deadline.checkpoint(stage="projection")
                    text = "\n".join(content.text[start:end] for start, end in spans)
                    sources, offset = [], 0
                    for start, end in spans:
                        for source in content.sources:
                            a, b = max(start, source.unit_start), min(end, source.unit_end)
                            if a < b:
                                source_start = source.source_start + a - source.unit_start
                                sources.append(replace(source, unit_start=offset + a - start,
                                    unit_end=offset + b - start, source_start=source_start, source_end=source_start + b - a))
                        offset += end - start + 1
                    unit_key = fingerprint({"source": content.key.stable_key, "spans": spans, "projection": projection_ref})
                    projected_key = replace(content.key, projection_ref=projection_ref)
                    # 同一对象多个片段保留独立 unit_key，证据键也绑定确切片段映射。
                    projected_key = replace(projected_key, object_key=unit_key)
                    context = " / ".join(content.heading_path)
                    unit = SearchUnit(projected_key, content.key, unit_key, content.kind, text, context, tuple(sources),
                        content.locators, content.provenance, document.allowed_view_ref, document.allowed_view_hash,
                        content.content_handle)
                    representations = []
                    for spec in profile.representations:
                        if spec.input_kind == "TEXT" and not (text or context):
                            continue
                        if spec.input_kind == "IMAGE" and content.content_handle is None:
                            continue
                        if self.model_gateway is None:
                            raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "未注入所需表示模型", stage="projection")
                        value = ModelInput("TEXT", text="\n".join(v for v in (context, text) if v),
                            input_role="DOCUMENT") if spec.input_kind == "TEXT" else ModelInput(
                                "IMAGE", image_handle=content.content_handle, input_role="DOCUMENT")
                        representation = await deadline.run(self.model_gateway.represent(value, spec), stage="projection_model")
                        representation.validate(spec, profile.max_representation_elements)
                        representations.append(representation)
                    count += 1
                    if count > profile.max_units:
                        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "检索单元超过预算", stage="projection")
                    pending.append(ProjectedUnit(unit, tuple(representations)))
                    if len(pending) >= profile.batch_size:
                        sequence += 1
                        yield ProjectionBatch(projection_ref, sequence, tuple(pending), fingerprint(encode(tuple(pending))))
                        pending = []
            if pending:
                sequence += 1
                yield ProjectionBatch(projection_ref, sequence, tuple(pending), fingerprint(encode(tuple(pending))))
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    def _chunks(self, content: AllowedContent, profile: ProjectionProfile) -> tuple[tuple[tuple[int, int], ...], ...]:
        context = " / ".join(content.heading_path)

        def fits(spans):
            text = "\n".join(content.text[a:b] for a, b in spans)
            count = self.tokenizer.count("\n".join(v for v in (context, text) if v))
            require(type(count) is int and count >= 0, "tokenizer 返回无效成本")
            return count <= profile.max_tokens

        whole = ((0, len(content.text)),) if content.text else ()
        if fits(whole):
            return (whole,) if content.text or content.content_handle else ()
        if content.kind == ContentKind.TABLE and content.table_rows:
            groups, current = [], (content.table_header,)
            for row in content.table_rows:
                if not fits((content.table_header, row)):
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "表头与单行超过预算，不能截断单位或单元格", stage="split")
                if not fits(current + (row,)):
                    groups.append(current)
                    current = (content.table_header,)
                current += (row,)
            if len(current) > 1:
                groups.append(current)
            return tuple(groups)
        if content.kind in {ContentKind.CODE, ContentKind.FORMULA, ContentKind.TABLE}:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "完整结构单元超过预算，需 App 提供可定位子结构", stage="split")
        # 只在句子/换行边界切分；超长单句不作无语义截断。
        ends = [m.end() for m in re.finditer(r"[。！？.!?](?:\s+|(?=[^\x00-\x7f])|$)|\n", content.text)]
        ends = sorted(set(ends + [len(content.text)]))
        chunks, start, previous = [], 0, 0
        for end in ends:
            if not fits(((start, end),)):
                if previous <= start:
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "单句超过预算，不能无来源截断", stage="split")
                chunks.append(((start, previous),))
                start = previous
                if not fits(((start, end),)):
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "单句超过预算", stage="split")
            previous = end
        if start < len(content.text):
            chunks.append(((start, len(content.text)),))
        return tuple(chunks)
