"""Adapters from model services to Knowledge Core ports.

KC reads model definitions through an HTTP DTO contract.  It does not import
the model service's ORM entity or repository, so the model service can later
move to its own database without changing KC persistence.
"""
import json
from hashlib import sha256
from typing import Sequence

from platform_core.dictionary import ModelCategory, Status
from platform_core.config.settings import get_embed_config
from knowledge_core.application.indexing import EmbeddingBatch, EmbeddingModelSnapshot
from platform_clients import AIModelClient, AIModelConfigClient


class AIModelEmbeddingGateway:
    """Use the shared embedding service without exposing it to Parser."""

    def __init__(self, client: AIModelClient | None = None):
        self._client = client or AIModelClient()

    async def embed_texts(
        self, *, model_key: str, texts: Sequence[str], is_query: bool = False,
    ) -> EmbeddingBatch:
        items = await self._client.call_embedding_model(
            model_name=model_key, texts=list(texts), is_query=is_query,
        )
        vectors = [list(item.embedding) for item in items]
        dimension = len(vectors[0]) if vectors else 0
        return EmbeddingBatch(vectors=vectors, model_key=model_key, dimension=dimension)


async def resolve_embedding_model(client: AIModelConfigClient, model_id: int) -> EmbeddingModelSnapshot:
    """Resolve and validate the Collection-bound model at INDEX start."""
    model = await client.get_model(model_id)
    if int(model.get("category") or 0) != int(ModelCategory.TXT_EMBEDDING):
        raise ValueError("Collection model is not a text embedding model")
    if int(model.get("status") or 0) != int(Status.ENABLED):
        raise ValueError("Collection embedding model is disabled")
    configured_dimension = get_embed_config().dimensions
    model_dimension = model.get("embedding_dimension")
    if model_dimension is None or configured_dimension is None:
        raise ValueError("embedding dimension is not configured")
    if int(model_dimension) != int(configured_dimension):
        raise ValueError("model embedding dimension does not match base.toml")
    model_key = str(model.get("model_name") or model.get("display_name") or "").strip()
    if not model_key:
        raise ValueError("embedding model has no technical name")
    fingerprint = sha256(json.dumps({
        "model_id": int(model.get("model_id") or model_id), "model_key": model_key,
        "provider": model.get("provider"), "dimension": int(model_dimension),
        "params": model.get("model_params") or {},
    }, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return EmbeddingModelSnapshot(
        model_id=int(model.get("model_id") or model_id), model_key=model_key,
        dimension=int(model_dimension), config_fingerprint=fingerprint,
    )
