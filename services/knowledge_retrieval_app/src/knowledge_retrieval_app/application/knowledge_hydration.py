"""从已发布清单构造有界检索视图；对象读取与模型操作不持有数据库事务。"""

from hashlib import sha256
import json
from types import MappingProxyType

from knowledge_algorithms.contracts.content import ContentKind, VisualAsset
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.identity import EvidenceKey, SourceSpan, canonical_json, fingerprint
from knowledge_algorithms.contracts.retrieval import EvidenceItem
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.projections.contracts import AllowedContent, ProjectedUnit, ProjectionProfile
from knowledge_algorithms.references.relations import ResolvedRelation
from knowledge_algorithms.references.citations import CitationTarget, document_citation_edges
from knowledge_algorithms.retrieval.exact import ExactIndex, IndexedUnit
from knowledge_algorithms.retrieval.published_batches import PublishedBatch, PublishedVectorBatches
from knowledge_algorithms.retrieval.published_metadata import PublishedMetadata
from knowledge_algorithms.retrieval.selection import PayloadCost
from ..adapters.knowledge_maxsim import prepare_maxsim
from ..domain.knowledge_ingestion import BINDING


async def hydrate_published(bundle, *, store, tokenizer, image_cost, limits):
    items, indexed, contents = {}, [], {}
    assets, text_payloads = {}, {}
    consumed = text_chars = 0
    vectors = PublishedVectorBatches(store=store, max_bytes=limits.max_object_bytes,
                                     max_elements=limits.max_vector_elements, prepare_multivector=prepare_maxsim)

    def account(count):
        nonlocal consumed
        consumed += count
        if consumed > limits.max_total_object_bytes:
            raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "已发布索引元数据超过读取预算", stage="index_metadata")

    async def read_json(handle, max_bytes=None):
        raw = await store.read(handle, max_bytes=min(max_bytes or limits.max_object_bytes, limits.max_object_bytes))
        account(len(raw))
        return json.loads(raw)

    for entry in bundle["assets"]:
        descriptor = decode(VisualAsset, entry["descriptor"])
        BINDING.check_owner(descriptor.key.owner_binding_ref)
        identity = (entry["handle"], descriptor.figure_key)
        require(identity not in assets, "同一逻辑图像资产被重复登记")
        assets[identity] = (descriptor, entry["handle"])

    def evidence(key, source_keys, kind, text, locators, provenance, view_ref, image_handle, spans=(), figure_key=None):
        raw = text.encode("utf-8")
        if image_handle is not None:
            identity = (image_handle, figure_key or key)
            require(identity in assets, "允许视图图像缺少对应逻辑图的已提交资产")
            asset, handle = assets[identity]
            digest, mime, byte_count = asset.sha256, asset.mime_type, asset.byte_count
            cost = PayloadCost(image_cost(asset), 1, byte_count)
        else:
            # 这里只保存允许视图，不从原文副本提取被策略删除的文字。
            if not raw:
                return None
            digest, byte_count, mime = sha256(raw).hexdigest(), len(raw), "text/plain"
            handle = "parsed-" + digest
            text_payloads[handle] = text
            cost = PayloadCost(tokenizer.count(text), 0, byte_count)
        return EvidenceItem(key, tuple(dict.fromkeys(source_keys)), kind, locators, provenance,
            view_ref, handle, digest, mime, byte_count, cost, 1, preview_handle=key.stable_key,
            source_spans=spans)

    for record in bundle["versions"]:
        manifest = await read_json(record["allowed_view_manifest_ref"])
        require(manifest["view_hash"] == record["allowed_view_hash"], "允许视图摘要与发布版本不一致")
        for entry in manifest["items"]:
            content = decode(AllowedContent, await read_json(entry["handle"], entry["max_bytes"]))
            require(content.key.stable_key == entry["key"]
                    and content.key.document_version_ref == record["version_id"]
                    and content.key.parse_run_ref == record["parse_run_id"], "允许视图对象身份错误")
            BINDING.check_owner(content.key.owner_binding_ref)
            require(content.key not in contents, "允许视图对象重复")
            contents[content.key] = content
            item = evidence(content.key, (content.key,), content.kind, content.text, content.locators,
                content.provenance, manifest["view_ref"], content.content_handle)
            if item is not None:
                items[item.key] = item
            if content.content_handle:
                asset, _ = assets[(content.content_handle, content.key)]
                require(asset.figure_key == content.key, "图像资产与允许节点不一致")
                image_item = evidence(asset.key, (content.key,), ContentKind.IMAGE, "", (asset.locator,),
                    asset.provenance, manifest["view_ref"], content.content_handle, figure_key=content.key)
                items[image_item.key] = image_item
        generation = record["generation_manifest"]
        profile = decode(ProjectionProfile, generation["profile"])
        batches = generation["batches"]
        require(len(batches) == generation["batch_count"]
                and generation["batch_digest"] == fingerprint([batches[str(i)]["hash"]
                    for i in range(1, len(batches) + 1)]), "发布投影清单摘要不一致")
        total = 0
        spaces = {spec.space_id for spec in profile.representations}
        for sequence in range(1, len(batches) + 1):
            entry = batches[str(sequence)]
            batch = PublishedBatch(entry["ref"], record["projection_ref"], sequence, entry["hash"],
                                   entry["count"], profile.max_representation_elements)
            members = {}
            for position, raw_unit in enumerate(await batch.read_units(store, limits.max_object_bytes)):
                unit_spaces = [rep["space_id"] for rep in raw_unit["representations"]]
                require(len(set(unit_spaces)) == len(unit_spaces) and set(unit_spaces) <= spaces
                        and not set(unit_spaces).intersection(raw_unit.get("skipped_image_spaces") or ()),
                        "投影出现重复、缺口重叠或未登记表示空间")
                thin = {**raw_unit, "representations": []}
                account(len(canonical_json(thin).encode("utf-8")))
                projected = decode(ProjectedUnit, thin)
                unit = projected.unit
                require(unit.source_key in contents and unit.key.owner_binding_ref == BINDING.binding_ref
                        and unit.key.document_version_ref == record["version_id"]
                        and unit.key.parse_run_ref == record["parse_run_id"]
                        and unit.key.projection_ref == record["projection_ref"]
                        and unit.allowed_view_hash == record["allowed_view_hash"]
                        and unit.allowed_view_ref == manifest["view_ref"], "投影单元越出允许范围")
                spans = tuple(SourceSpan(s.source_key.object_key, s.source_start, s.source_end) for s in unit.sources)
                sources = (unit.source_key,) + tuple(s.source_key for s in unit.sources)
                item = evidence(unit.key, sources, unit.kind, unit.text, unit.locators,
                    unit.provenance, unit.allowed_view_ref, unit.content_handle, spans, figure_key=unit.source_key)
                if item is not None:
                    require(unit.key not in items, "投影证据身份重复")
                    items[unit.key] = item
                    attributes = tuple((field, record[field]) for field in
                                       ("title", "business_asset_ref", "created_at"))
                    image_hash = assets[(unit.content_handle, unit.source_key)][0].sha256 if unit.content_handle else None
                    value = IndexedUnit(projected, attributes, image_hash)
                    indexed.append(value)
                    text_chars += len(unit.text) + len(unit.context)
                    if text_chars > limits.max_text_chars:
                        raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "已发布索引文本超过内存预算",
                                             stage="index_metadata")
                    for space in unit_spaces:
                        members.setdefault(space, []).append((position, value))
                total += 1
                if len(indexed) > limits.max_units:
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "已发布单元超过有界精确索引预算",
                                         stage="index_metadata")
            raw_unit = None
            vectors.add(batch, members)
            if len(items) > limits.max_units * 4:
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "规范证据对象超过读取预算", stage="index_metadata")
        require(total == generation["unit_count"], "发布单元总数不一致")
    relations = tuple(decode(ResolvedRelation, raw) for raw in bundle["relations"]
                      if raw.get("relation_type") in {"REFERS_TO", "CAPTION_OF", "HAS_ASSET", "PART_OF", "CITES_DOCUMENT"})
    relations = tuple(edge for edge in relations if edge.source_key in items and edge.target_key in items)
    catalog = tuple(CitationTarget(decode(EvidenceKey, record["citation_target"]["key"]),
        tuple(record["citation_target"]["identifiers"])) for record in bundle["versions"] if record.get("citation_target"))
    catalog = tuple(target for target in catalog if target.key in items)
    cross_edges = document_citation_edges(tuple(contents.values()), catalog, policy_revision="exact-citation/v2")
    require(len(relations) + len(cross_edges) <= limits.max_relations, "精确跨文档引用超过预算")
    relations = tuple({edge.relation_key: edge for edge in relations + cross_edges}.values())
    index = ExactIndex(indexed, max_units=limits.max_units, max_text_chars=limits.max_text_chars,
                       max_vector_elements=limits.max_vector_elements)
    # 计入展平元数据以外的分词统计与证据对象；向量不进入缓存。
    estimated_bytes = (consumed + sum(len(terms) for terms in index.terms.values()) * 96
                       + len(index.frequency) * 96 + len(items) * 384 + len(relations) * 384)
    return PublishedMetadata(index, MappingProxyType(items), relations, vectors,
                             MappingProxyType(text_payloads), estimated_bytes)
