"""Small model-service response contracts with no provider dependencies."""

from pydantic import BaseModel, Field


class EmbeddingDataItem(BaseModel):
    """One vector returned by the embedding service."""

    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Index position in the batch")
    object: str = Field("embedding", description="Object type")
