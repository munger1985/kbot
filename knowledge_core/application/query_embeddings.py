"""Collection-bound query embedding contract.

Query vectors are generated at retrieval time from the same model identity used
by the Collection's INDEX jobs.  Callers never submit an arbitrary vector as a
substitute for this resolution step.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from knowledge_core.application.indexing import EmbeddingGateway, EmbeddingModelSnapshot
    from knowledge_core.persistence import KnowledgeCoreUnitOfWork


class QueryEmbeddingProvider(Protocol):
    async def embed_for_collections(
        self, *, query: str, collection_ids: Sequence[UUID],
    ) -> dict[UUID, list[float]]: ...


class CollectionQueryEmbeddingProvider:
    """Group Collection queries by their configured model before embedding."""

    def __init__(
        self, *,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
        embedding_gateway: EmbeddingGateway,
        model_resolver: Callable[[UUID], Awaitable[EmbeddingModelSnapshot]],
    ):
        self._uow_factory = uow_factory
        self._embedding_gateway = embedding_gateway
        self._model_resolver = model_resolver

    async def embed_for_collections(
        self, *, query: str, collection_ids: Sequence[UUID],
    ) -> dict[UUID, list[float]]:
        if not query.strip() or not collection_ids:
            return {}
        model_ids: dict[UUID, UUID] = {}
        async with self._uow_factory() as uow:
            if uow.collections is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            for collection_id in sorted(set(collection_ids)):
                collection = await uow.collections.get_by_id(collection_id=collection_id)
                if collection is None:
                    raise ValueError(f"collection not found: {collection_id}")
                model_ids[collection_id] = collection.embedding_model_id

        snapshots: dict[UUID, EmbeddingModelSnapshot] = {}
        for model_id in sorted(set(model_ids.values()), key=str):
            snapshot = await self._model_resolver(model_id)
            snapshot.validate()
            snapshots[model_id] = snapshot

        by_model: dict[UUID, list[UUID]] = {}
        for collection_id, model_id in model_ids.items():
            by_model.setdefault(model_id, []).append(collection_id)
        vectors: dict[UUID, list[float]] = {}
        from knowledge_core.application.indexing import validate_embedding_batch
        for model_id, scoped_collections in by_model.items():
            model = snapshots[model_id]
            batch = await self._embedding_gateway.embed_texts(
                served_model_name=model.served_model_name, texts=[query], is_query=True,
            )
            validate_embedding_batch(batch=batch, model=model, expected_count=1)
            for collection_id in scoped_collections:
                vectors[collection_id] = list(batch.vectors[0])
        return vectors
