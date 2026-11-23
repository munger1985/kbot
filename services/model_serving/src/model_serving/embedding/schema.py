from typing import Literal

from uuid import UUID

from pydantic import BaseModel, Field


class DenseRepresentationRequest(BaseModel):
    """有界表示请求；模型空间由服务配置而非调用方覆盖。"""
    model_id: UUID
    text: str = Field(min_length=1, max_length=16000)
    input_role: Literal["QUERY", "DOCUMENT"]
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_config = {"extra": "forbid"}


# Define embedding request model
class EmbeddingRequest(BaseModel):
    """Embedding request parameter model."""
    
    model_id: UUID
    texts: list[str] = Field(..., description="List of texts to generate embeddings for")
    batch_size: int | None = Field(32, description="Batch processing size")
    is_query: bool = Field(True, description="Whether the texts are query inputs")

class SimilarityRequest(BaseModel):
    """Similarity calculation request parameter model."""
    model_id: UUID
    text1: str = Field(..., description="First text string")
    text2: str = Field(..., description="Second text string")
    method: str = Field("cosine", description="Similarity calculation method: supports 'cosine' (cosine similarity) and 'dot' (dot product)")


class OpenAIEmbeddingRequest(BaseModel):
    """OpenAI 兼容的文本向量请求。"""

    model: UUID
    input: str | list[str]
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = Field(default=None, gt=0)
