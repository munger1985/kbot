"""直接检索首批历史；不参与 Agent、模型调用或召回。"""

from dataclasses import replace
from datetime import timedelta
import json
from uuid import UUID

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.retrieval import EvidenceSet
from knowledge_algorithms.contracts.wire import decode


RETENTION = timedelta(days=30)


def saved_conditions(request):
    # 显式白名单：不保存原始图片、base64、查询图片句柄或其摘要。
    return dict(text=(request.text or "").strip() or None, image_count=len(request.images),
        image_intent=(request.image_intent or "SOURCE") if len(request.images) == 1 and not request.text else None,
        requested_count=request.requested_count, output_types=list(request.output_types),
        combination=request.combination, mode=request.mode,
        predicates=[item.model_dump(mode="json") for item in request.predicates],
        order_by=[item.model_dump(mode="json") for item in request.order_by])


def first_batch(evidence, allowed):
    """只投影首批主命中及其关联对象，不沿用任何游标。"""
    hits = tuple(hit for hit in evidence.primary_hits if hit.evidence_key in allowed)
    primary = {hit.evidence_key for hit in hits}
    groups, visible = [], set(primary)
    for group in evidence.groups:
        if group.anchor_key not in primary:
            continue
        links = tuple(key for key in group.linked_keys if key in allowed)
        endpoints = {group.anchor_key, *links}
        groups.append(replace(group, linked_keys=links, continuation=None, relations=tuple(
            edge for edge in group.relations if {edge.source_key, edge.target_key} <= endpoints)))
        visible.update(links)
    original_primary = {hit.evidence_key for hit in evidence.primary_hits}
    original = original_primary | {key for group in evidence.groups if group.anchor_key in original_primary
                                   for key in group.linked_keys}
    partial = not original <= visible
    return replace(evidence, primary_hits=hits, groups=tuple(groups), continuation=None,
        evidence=tuple(item for item in evidence.evidence if item.key in visible),
        status="PARTIAL" if partial else evidence.status,
        reason="HISTORY_EVIDENCE_CHANGED" if partial else evidence.reason,
        context_selection=replace(evidence.context_selection, selected_evidence_keys=()),
        matches=(), trace=()), partial


class KnowledgeSearchHistory:
    def __init__(self, uow_factory, retrieval):
        self.uow_factory, self.retrieval = uow_factory, retrieval

    async def save(self, access, library_id, request, receipt):
        require(receipt.manifest_ref, "检索历史缺少不可变证据清单")
        now = self.retrieval.clock()
        async with self.uow_factory() as uow:
            row = await uow.search_history(access).add(dict(library_id=library_id,
                subject_ref=access.subject_ref, conditions_json=saved_conditions(request),
                manifest_ref=receipt.manifest_ref, result_count=len(receipt.evidence_set.primary_hits),
                created_at=now, expires_at=now + RETENTION))
            identity = row.history_id
            await uow.commit()
        return identity

    async def page(self, access, *, limit, offset):
        async with self.uow_factory() as uow:
            entries, more = await uow.search_history(access).page(now=self.retrieval.clock(),
                                                                limit=limit, offset=offset)
            libraries = await uow.library(access).listing()
        labels = {str(library["library_id"]): library["display_name"] for library in libraries}
        return dict(items=[dict(entry, library_name=labels.get(str(entry["library_id"])))
                           for entry in entries], has_more=more)

    async def delete(self, access, identity):
        async with self.uow_factory() as uow:
            await uow.search_history(access).remove(identity, now=self.retrieval.clock())
            await uow.commit()

    async def record(self, access, identity):
        async with self.uow_factory() as uow:
            repo = uow.search_history(access)
            row = await repo.owned(identity, now=self.retrieval.clock())
            require(row is not None, "检索历史不存在或已过期")
            return dict(repo.view(row), manifest_ref=row.manifest_ref)

    async def current(self, access, record, items):
        library_id = record["library_id"]
        refs = {UUID(item.key.document_version_ref) for item in items}
        async with self.uow_factory() as uow:
            name, members = await uow.library(access).history_versions(library_id, refs)
            versions = await uow.knowledge_retrieval.passage_versions(
                access, members, allow_missing=True)
        # 发布行保存摘要；证据使用允许视图清单的 allowed-<摘要> 引用。
        allowed = {item.key for item in items if
                   (version := versions.get(item.key.document_version_ref))
                   and version["view_hash"] and item.allowed_view_ref == "allowed-" + version["view_hash"]
                   and version["parse_run_id"] == item.key.parse_run_ref}
        return name, allowed

    async def load_batch(self, record):
        raw = await self.retrieval.store.read(record["manifest_ref"], max_bytes=self.retrieval.limits.max_object_bytes)
        evidence = decode(EvidenceSet, json.loads(raw))
        return first_batch(evidence, {item.key for item in evidence.evidence})[0]

    async def replay(self, access, record):
        # 在读取不可变清单前先验证库权限，即使首批没有任何命中。
        await self.current(access, record, ())
        evidence = await self.load_batch(record)
        name, allowed = await self.current(access, record, evidence.evidence)
        batch, partial = first_batch(evidence, allowed)
        return name, batch, partial

    async def reauthorize(self, access, record, items):
        # 长流式读取期间同样检查历史是否删除或到期，以及来源是否仍可读。
        try:
            await self.record(access, record["history_id"])
            name, allowed = await self.current(access, record, items)
        except AlgorithmError as exc:
            if exc.code != ErrorCode.INVALID_CONTRACT:
                raise
            raise AlgorithmError(ErrorCode.SCOPE_CHANGED, "检索历史已过期、删除或不可读", stage="scope") from exc
        if not {item.key for item in items} <= allowed:
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE, "历史证据已撤回、换版或不可读", stage="snapshot")
        return name
