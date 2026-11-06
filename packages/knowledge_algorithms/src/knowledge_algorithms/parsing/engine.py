"""中立解析编排：源校验、区域修复、结构及显式关系、同一事件流输出。"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import AsyncIterator

from ..contracts.content import ContentKind, ContentNode, Page, ProvenanceKind, VisualAsset
from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import AppBackendBinding, fingerprint
from ..contracts.wire import decode, encode
from ..execution import Deadline
from ..references.figures import FigureIdentity, FigureResolution, ReferenceOccurrence, extract_figure_references, resolve_figure_reference
from ..references.relations import RelationType, ResolvedRelation
from .contracts import ExtractedRegion, ParseProfile, ParsingAdapter, RepairAdapter, SourceReader
from .events import ParseEvent, ParseEventType, ParseRunIdentity, ParseStreamValidator, local_key
from .structure import order_and_structure


_VISUAL = {ContentKind.FIGURE, ContentKind.PHOTO, ContentKind.IMAGE, ContentKind.PAGE, ContentKind.REGION}


def quality(region: ExtractedRegion) -> tuple[float, float, bool]:
    texts = [n.text for n in region.nodes if n.kind not in _VISUAL]
    text = "".join(texts)
    bad = sum(char == "\ufffd" or ord(char) < 32 and not char.isspace() for char in text)
    return bad / max(1, len(text)), region.text_coverage, bool(text.strip()) or not texts


def passes(region: ExtractedRegion, profile: ParseProfile) -> bool:
    bad, coverage, usable = quality(region)
    visual_only = all(n.kind in _VISUAL for n in region.nodes)
    return usable and (visual_only or bad <= profile.max_bad_char_ratio and coverage >= profile.min_region_coverage)


class VerifiedReader:
    def __init__(self, source: SourceReader, limit: int):
        self.descriptor = source.descriptor
        self.source = source
        self.limit = limit
        self.complete = False
        self.started = False

    async def chunks(self) -> AsyncIterator[bytes]:
        require(not self.started, "源读取只能消费一次")
        self.started = True
        digest, size = sha256(), 0
        async for block in self.source.chunks():
            require(type(block) is bytes and bool(block), "源流必须提供非空字节块")
            size += len(block)
            if size > min(self.limit, self.descriptor.byte_count):
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "源流超过冻结大小", stage="source")
            digest.update(block)
            yield block
        require(size == self.descriptor.byte_count and digest.hexdigest() == self.descriptor.sha256,
                "源内容大小或摘要不匹配", stage="source")
        self.complete = True


@dataclass(frozen=True, slots=True)
class RepairAlternative:
    region_key: str
    nodes: tuple[ContentNode, ...]
    reason: str
    adopted: bool


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    run: ParseRunIdentity
    pages: tuple[Page, ...]
    nodes: tuple[ContentNode, ...]
    assets: tuple[VisualAsset, ...]
    references: tuple[ReferenceOccurrence, ...]
    relations: tuple[ResolvedRelation, ...]
    quality_status: str
    manifest_hash: str
    resolutions: tuple[FigureResolution, ...]
    alternatives: tuple[RepairAlternative, ...]
    warnings: tuple[str, ...]
    failed_regions: tuple[str, ...]


class ParseEngine:
    def __init__(self, binding: AppBackendBinding, adapters: tuple[ParsingAdapter, ...],
                 repair_adapter: RepairAdapter | None = None, *, artifact_provider=None):
        self.binding, self.adapters, self.repair_adapter = binding, adapters, repair_adapter
        self.artifact_provider = artifact_provider
        mime_types = [mime for adapter in adapters for mime in adapter.mime_types]
        require(len(set(mime_types)) == len(mime_types), "格式路由必须在组合根唯一登记")

    async def parse_events(self, source: SourceReader, profile: ParseProfile,
                           run: ParseRunIdentity) -> AsyncIterator[ParseEvent]:
        self.binding.check_owner(run.owner_binding_ref)
        self.binding.check_owner(source.descriptor.owner_binding_ref)
        require(source.descriptor.sha256 == run.source_sha256 and profile.profile_hash == run.profile_hash,
                "解析输入或 Profile 与冻结任务不一致")
        selected = [adapter for adapter in self.adapters if source.descriptor.mime_type in adapter.mime_types]
        if not selected:
            raise AlgorithmError(ErrorCode.UNSUPPORTED_INPUT, "没有受支持的格式适配器", stage="route")
        adapter = selected[0]
        if source.descriptor.byte_count > profile.max_input_bytes:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "输入大小超过预算", stage="source")
        deadline = Deadline(profile.timeout_seconds)
        validator = ParseStreamValidator(run, max_events=profile.max_objects * 4 + 4,
                                        max_objects=profile.max_objects, max_payload_bytes=profile.max_event_bytes)

        def event(kind, payload, dependencies=()):
            sequence = validator.event_count + 1
            artifacts = self.artifact_provider(kind, payload) if self.artifact_provider is not None else ()
            result = ParseEvent.create(run=run, sequence=sequence, batch_key=f"batch-{sequence}",
                                       event_type=kind, payload=payload, dependency_keys=tuple(dependencies),
                                       artifact_handles=tuple(artifacts))
            validator.accept(result)
            return result

        yield event(ParseEventType.RUN_STARTED, {"profile_hash": profile.profile_hash, "parser_revision": adapter.revision})
        verified = VerifiedReader(source, profile.max_input_bytes)
        iterator = adapter.extract(verified, profile, run)
        nodes, assets, pages, alternatives, failed_regions = [], [], [], [], []
        text_bytes = 0

        def account(batch_nodes):
            nonlocal text_bytes
            text_bytes += sum(len(node.text.encode("utf-8")) for node in batch_nodes)
            if text_bytes > profile.max_text_bytes:
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "文本及修复候选超过内存载荷预算", stage="parse")

        def check_nodes(batch_nodes):
            for node in batch_nodes:
                self.binding.check_owner(node.key.owner_binding_ref)
                require(node.key.document_version_ref == source.descriptor.document_version_ref
                        and node.key.parse_run_ref == run.parse_run_ref, "提取节点版本或解析代次不匹配")
                require(node.key.object_key == local_key(algorithm_version=profile.algorithm_version,
                        native_ref=node.native_ref, kind=node.kind.value), "提取节点未使用规范本地身份")

        try:
            while True:
                try:
                    batch = await deadline.run(anext(iterator), stage="extract")
                except StopAsyncIteration:
                    break
                pages.extend(batch.pages)
                assets.extend(batch.assets)
                if len(pages) > profile.max_pages:
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "页面数超过预算", stage="parse")
                for region in batch.regions:
                    check_nodes(region.nodes)
                    account(region.nodes)
                    active = region
                    if not passes(active, profile):
                        alternatives.append(RepairAlternative(region.region_key, region.nodes, "ORIGINAL", False))
                    for attempt in range(profile.max_repair_attempts):
                        if passes(active, profile) or self.repair_adapter is None or not profile.repair_strategies:
                            break
                        strategy = profile.repair_strategies[min(attempt, len(profile.repair_strategies) - 1)]
                        candidate = await deadline.run(self.repair_adapter.repair(
                            active, strategy=strategy, profile=profile, run=run), stage="repair")
                        check_nodes(candidate.nodes)
                        account(candidate.nodes)
                        require(candidate.region_key == region.region_key and
                                {n.key for n in candidate.nodes} == {n.key for n in active.nodes}, "局部修复不能更换对象身份")
                        expected = ProvenanceKind(strategy)
                        require(all(n.provenance.kind == expected for n in candidate.nodes), "修复输出来源与任务不一致")
                        old, new = quality(active), quality(candidate)
                        improved = new[2] and new[0] <= old[0] and new[1] >= old[1] and new != old
                        alternatives.append(RepairAlternative(region.region_key, candidate.nodes,
                                            "IMPROVED" if improved else "NO_IMPROVEMENT", improved))
                        if not improved:
                            break
                        active = candidate
                    if not passes(active, profile):
                        failed_regions.append(region.region_key)
                    nodes.extend(active.nodes)
                    if len(nodes) + len(assets) + len(pages) > profile.max_objects:
                        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "解析对象数超过预算", stage="parse")
                await deadline.checkpoint(stage="parse")
            require(verified.complete, "解析器未完成源内容校验", stage="source")
            require(bool(nodes), "解析没有产生可定位内容", stage="parse")
            require(len({p.page_no for p in pages}) == len(pages), "页面序号重复")
            page_numbers = {p.page_no for p in pages}
            require(all(loc.page_no is None or loc.page_no in page_numbers for node in nodes for loc in node.locators),
                    "节点引用了不存在的页面")
            structured, warnings = order_and_structure(tuple(nodes), profile.layout)
            figures = tuple(FigureIdentity(n.key, n.display_label, n.numbering_scope or "document", n.native_anchor)
                            for n in structured if n.kind == ContentKind.FIGURE and n.display_label is not None)
            occurrences, relations = [], []
            for node in structured:
                if node.kind not in {ContentKind.PARAGRAPH, ContentKind.LIST_ITEM, ContentKind.FOOTNOTE}:
                    continue
                refs = extract_figure_references(node.text, node.key, numbering_scope=node.numbering_scope,
                                                policy_version=profile.algorithm_version)
                for ref in refs:
                    resolution = resolve_figure_reference(ref, figures)
                    occurrences.append(resolution)
                    if resolution.target_key is not None:
                        relations.append(ResolvedRelation(fingerprint([ref.occurrence_key, resolution.target_key.stable_key]),
                            RelationType.REFERS_TO, node.key, resolution.target_key, resolution.basis,
                            ref.occurrence_key, ref.span))
            node_by_key = {n.key.object_key: n for n in structured}
            for node in structured:
                if node.kind == ContentKind.CAPTION and node.parent_key in node_by_key:
                    parent = node_by_key[node.parent_key]
                    if parent.kind == ContentKind.FIGURE:
                        relations.append(ResolvedRelation(fingerprint([node.key.stable_key, "caption"]),
                            RelationType.CAPTION_OF, node.key, parent.key, "EXPLICIT_PARENT"))
                elif (node.kind in {ContentKind.FIGURE, ContentKind.REGION} or node.provenance.kind == ProvenanceKind.OCR) and node.parent_key in node_by_key:
                    parent = node_by_key[node.parent_key]
                    if parent.kind in {ContentKind.FIGURE, ContentKind.PAGE}:
                        relations.append(ResolvedRelation(fingerprint([node.key.stable_key, "part"]),
                            RelationType.PART_OF, node.key, parent.key,
                            "EXPLICIT_TRANSCRIPTION_REGION" if node.provenance.kind == ProvenanceKind.OCR else "EXPLICIT_PARENT"))
            for asset in assets:
                self.binding.check_owner(asset.key.owner_binding_ref)
                require(asset.figure_key in {n.key for n in structured}, "原图缺少对应逻辑对象")
                relations.append(ResolvedRelation(fingerprint([asset.key.stable_key, "asset"]), RelationType.HAS_ASSET,
                                                  asset.figure_key, asset.key, "EXPLICIT_ASSET"))
            items = [(ParseEventType.PAGE_BATCH, [encode(p) for p in pages]),
                     (ParseEventType.NODE_BATCH, [{"local_key": n.key.object_key, **encode(n)} for n in structured]),
                     (ParseEventType.ASSET_BATCH, [{"local_key": a.key.object_key, **encode(a)} for a in assets]),
                     (ParseEventType.REFERENCE_BATCH, [{"local_key": resolution.occurrence.occurrence_key,
                                                       **encode(resolution)} for resolution in occurrences])]
            for kind, objects in items:
                for index in range(0, len(objects), profile.batch_size):
                    await deadline.checkpoint(stage="emit")
                    yield event(kind, {"objects": objects[index:index + profile.batch_size]})
            status = "COMPLETE" if not failed_regions else "PARTIAL" if profile.allow_partial else "FAILED"
            yield event(ParseEventType.STRUCTURE_FINALIZED,
                        {"relations": encode(tuple(relations)), "alternatives": encode(tuple(alternatives)),
                         "warnings": list(warnings)},
                        {k.object_key for r in relations for k in (r.source_key, r.target_key)})
            yield event(ParseEventType.RUN_COMPLETED, {"counts": validator.object_counts,
                        "event_count": validator.event_count, "ordered_digest": validator.ordered_digest,
                        "quality_report": {"status": status, "failed_regions": failed_regions}})
        except AlgorithmError as exc:
            yield event(ParseEventType.RUN_FAILED, {"error_code": exc.code.value, "stage": exc.stage})
        except Exception:
            yield event(ParseEventType.RUN_FAILED, {"error_code": "PARSE_QUALITY_INSUFFICIENT", "stage": "adapter"})
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def parse(self, source: SourceReader, profile: ParseProfile, run: ParseRunIdentity) -> ParsedDocument:
        pages, nodes, assets, refs, relations, resolutions, alternatives, warnings = [], [], [], [], [], [], [], []
        manifest = None
        async for event in self.parse_events(source, profile, run):
            payload = json.loads(event.payload_json)
            if event.event_type == ParseEventType.RUN_FAILED:
                raise AlgorithmError(ErrorCode(payload["error_code"]), "解析明确失败", stage=payload["stage"])
            contract_and_list = {ParseEventType.PAGE_BATCH: (Page, pages), ParseEventType.NODE_BATCH: (ContentNode, nodes),
                                 ParseEventType.ASSET_BATCH: (VisualAsset, assets)}
            if event.event_type in contract_and_list:
                contract, result = contract_and_list[event.event_type]
                for item in payload["objects"]:
                    fields = {k: v for k, v in item.items() if k != "local_key" or contract is Page}
                    result.append(decode(contract, fields))
            elif event.event_type == ParseEventType.REFERENCE_BATCH:
                for item in payload["objects"]:
                    resolution = decode(FigureResolution, {k: v for k, v in item.items() if k != "local_key"})
                    resolutions.append(resolution)
                    refs.append(resolution.occurrence)
            elif event.event_type == ParseEventType.STRUCTURE_FINALIZED:
                relations.extend(decode(ResolvedRelation, item) for item in payload["relations"])
                alternatives.extend(decode(RepairAlternative, item) for item in payload["alternatives"])
                warnings.extend(payload["warnings"])
            elif event.event_type == ParseEventType.RUN_COMPLETED:
                manifest = payload
        require(manifest is not None, "解析流缺少完成清单")
        return ParsedDocument(run, tuple(pages), tuple(nodes), tuple(assets), tuple(refs), tuple(relations),
                              manifest["quality_report"]["status"], fingerprint(manifest), tuple(resolutions),
                              tuple(alternatives), tuple(warnings), tuple(manifest["quality_report"]["failed_regions"]))
