"""Adapters from model services to Knowledge Core ports.

KC reads model definitions through an HTTP DTO contract.  It does not import
the model service's ORM entity or repository, so the model service can later
move to its own database without changing KC persistence.
"""
import json
from hashlib import sha256
from typing import Sequence
from uuid import UUID

from platform_core.dictionary import ModelCategory, Status
from knowledge_core.application.indexing import EmbeddingBatch, EmbeddingModelSnapshot
from platform_clients import AIModelClient, AIModelConfigClient


class AIModelEmbeddingGateway:
    """Use the shared embedding service without exposing it to Parser."""

    def __init__(self, client: AIModelClient):
        self._client = client

    async def embed_texts(
        self, *, served_model_name: str, texts: Sequence[str],
        is_query: bool = False,
    ) -> EmbeddingBatch:
        items = await self._client.call_embedding_model(
            served_model_name=served_model_name, texts=list(texts), is_query=is_query,
        )
        vectors = [list(item.embedding) for item in items]
        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingBatch(
            vectors=vectors, served_model_name=served_model_name,
            dimension=dimension,
        )


async def resolve_embedding_model(
    client: AIModelConfigClient,
    model_id: UUID,
    *,
    expected_dimension: int,
) -> EmbeddingModelSnapshot:
    """Resolve and validate the Collection-bound model at INDEX start."""
    model = await client.get_model(model_id)
    if int(model.get("category") or 0) != int(ModelCategory.TXT_EMBEDDING):
        raise ValueError("Collection model is not a text embedding model")
    if int(model.get("status") or 0) != int(Status.ENABLED):
        raise ValueError("Collection embedding model is disabled")
    model_params = model.get("model_params") or {}
    model_dimension = model_params.get("embedding_dimension")
    if model_dimension is None:
        raise ValueError("embedding dimension is not configured")
    if int(model_dimension) != int(expected_dimension):
        raise ValueError(
            "模型维度与 Knowledge Core 配置的向量维度不一致"
        )
    served_model_name = str(model.get("served_model_name") or "").strip()
    if not served_model_name:
        raise ValueError("embedding model has no served_model_name")
    resolved_model_id = UUID(str(model.get("model_id") or model_id))
    fingerprint = sha256(json.dumps({
        "model_id": str(resolved_model_id),
        "served_model_name": served_model_name,
        "provider_model_name": model.get("provider_model_name"),
        "provider": model.get("provider"), "dimension": int(model_dimension),
        "params": model.get("model_params") or {},
    }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return EmbeddingModelSnapshot(
        model_id=resolved_model_id,
        served_model_name=served_model_name, dimension=int(model_dimension),
        config_fingerprint=fingerprint,
    )
