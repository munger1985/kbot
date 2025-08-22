from typing import Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Configuration for reranker models."""
    model_name: str = Field(..., description="Name of the reranker model")
    provider: str = Field(..., description="Provider of the reranker model")
    max_tokens: int | None = Field(512, description="Maximum input sequence length")

class BaseReranker(ABC):
    """Abstract base class for reranker models."""
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize the reranker model."""
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health check for a remote or local model"""
        pass