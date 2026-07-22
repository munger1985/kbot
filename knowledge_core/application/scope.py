"""Domain/Agent/Collection authorization boundary for KC retrieval."""

from collections.abc import Callable, Sequence

from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class KnowledgeScopeError(ValueError):
    """The requested retrieval scope is not authorized or no longer active."""


class KnowledgeCoreScopeService:
    def __init__(self, *, app_id: int, uow_factory: Callable[[], KnowledgeCoreUnitOfWork]):
        self._app_id = app_id
        self._uow_factory = uow_factory

    async def resolve_agent_collections(
        self, *, domain_id: int, agent_id: int, collection_ids: Sequence[int],
    ) -> tuple[int, ...]:
        requested = tuple(sorted(set(int(value) for value in collection_ids)))
        if domain_id <= 0 or agent_id <= 0 or not requested:
            raise KnowledgeScopeError("domain_id, agent_id and collection_ids are required")
        async with self._uow_factory() as uow:
            if uow.collections is None or uow.bindings is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            active: list[int] = []
            for collection_id in requested:
                collection = await uow.collections.get_by_id_scope(
                    app_id=self._app_id, domain_id=domain_id, collection_id=collection_id,
                )
                if collection is None:
                    raise KnowledgeScopeError("collection is outside the Domain scope")
                binding = await uow.bindings.get_by_consumer_collection(
                    consumer_type="AGENT", consumer_id=str(agent_id), collection_id=collection_id,
                )
                if binding is None or binding.status != "ACTIVE":
                    raise KnowledgeScopeError("Agent is not authorized for Collection")
                # Binding remains intact while a Collection is disabled, but
                # disabled data is skipped rather than returned by retrieval.
                if collection.status == "ACTIVE":
                    active.append(collection_id)
            return tuple(active)
