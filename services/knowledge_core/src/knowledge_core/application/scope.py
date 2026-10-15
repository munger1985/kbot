"""KC 检索的 Domain、Agent 与 Collection 边界。"""

from collections.abc import Callable, Sequence
from uuid import UUID

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeScopeError(ValueError):
    """请求的检索范围无效或已停用。"""


class KnowledgeCoreScopeService:
    def __init__(self, *, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._uow_factory = uow_factory

    async def resolve_agent_collections(
        self, *, domain_id: int, agent_id: UUID, collection_ids: Sequence[UUID],
    ) -> tuple[UUID, ...]:
        requested = tuple(sorted(set(collection_ids)))
        if domain_id <= 0 or not requested:
            raise KnowledgeScopeError("domain_id, agent_id and collection_ids are required")
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            active: list[UUID] = []
            for collection_id in requested:
                collection = await uow.collections.get_by_id_scope(
                    domain_id=domain_id,
                    collection_id=collection_id,
                )
                if collection is None:
                    raise KnowledgeScopeError("collection is outside the Domain scope")
                binding = await uow.bindings.get_by_consumer_collection(
                    consumer_type="AGENT", consumer_id=agent_id, collection_id=collection_id,
                )
                if binding is None or binding.status != "ACTIVE":
                    raise KnowledgeScopeError("Agent is not authorized for Collection")
                # Binding remains intact while a Collection is disabled, but
                # disabled data is skipped rather than returned by retrieval.
                if collection.status == "ACTIVE":
                    active.append(collection_id)
            return tuple(active)

    async def validate_revision_scope(
        self,
        *,
        collection_ids: Sequence[UUID],
        bundle_revision_ids: Sequence[UUID],
    ) -> tuple[UUID, ...]:
        """Revision 必须精确属于已授权 Collection 且仍是当前可检索版本。"""
        requested = tuple(
            sorted(set(bundle_revision_ids), key=lambda item: item.bytes)
        )
        if not requested:
            raise KnowledgeScopeError("bundle_revision_ids are required")
        async with self._uow_factory() as uow:
            if uow.revisions is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            resolved = await uow.revisions.resolve_current_scope(
                collection_ids=list(collection_ids),
                bundle_revision_ids=list(requested),
            )
        if set(resolved) != set(requested):
            raise KnowledgeScopeError(
                "one or more Bundle Revisions are outside the exact scope"
            )
        return requested

    async def resolve_current_revision_scope(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        collection_ids: Sequence[UUID],
    ) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
        """按 Agent 授权解析当前可检索的不可变 Revision 快照。"""
        scoped_collections = await self.resolve_agent_collections(
            domain_id=domain_id,
            agent_id=agent_id,
            collection_ids=collection_ids,
        )
        if not scoped_collections:
            raise KnowledgeScopeError("no active Collection is available")
        async with self._uow_factory() as uow:
            if uow.revisions is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            revision_ids = await uow.revisions.list_current_scope(
                collection_ids=list(scoped_collections)
            )
        if not revision_ids:
            raise KnowledgeScopeError("no current Bundle Revision is available")
        return scoped_collections, tuple(revision_ids)
