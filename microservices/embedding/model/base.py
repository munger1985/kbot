from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram


class EmbeddingConfig(BaseModel):
    model_name: str
    provider: str
    max_tokens: int

class LocalEmbeddingConfig(EmbeddingConfig):
    model_path: str | None = None
    device: str | None = None
    device_map: str | None = None
    max_memory: int | None = None
    trust_remote_code: bool = False
    use_fp16: bool = False
    local_files_only: bool = False
    compile_model: bool = True # True when PyTorch 2.0+ else False

class RemoteEmbeddingConfig(EmbeddingConfig):
    api_key: str
    endpoint: str
    timeout: int = 30
    max_retries: int = 3
    organization: str
    deployment_name: str
    api_version: str = "2023-05-15"
    additional_params: dict = {}

class EmbeddingDataItem(BaseModel):
    embedding: list[float] = Field(..., description="The embedding vector.")
    index: int = Field(..., description="The index of the embedding in the batch.")
    object: str = Field("embedding", description="The object type, always 'embedding'.")

class EmbeddingResponse(BaseModel):
    data: list[EmbeddingDataItem] = Field(..., description="List of embedding data items.")
    model: str = Field(..., description="Embedding model name used.")
    object: str = Field("list", description="The object type, always 'list'.")
    usage: dict[str, int] = Field(..., description="Token usage information.")

class BaseEmbedding(ABC):
    LATENCY_HIST = Histogram(
        'embedding_latency_seconds', 
        'Embedding latency in seconds',
        ['model_type']
    )
    ERROR_COUNTER = Counter(
        'embedding_errors_total', 
        'Total number of embedding errors', 
        ['provider']
    )

    @abstractmethod
    async def startup(self) -> None:
        """Initialize the embedding model and create client"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the embedding model and client"""
        pass
    
    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for a list of texts in OpenAI standard format.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResponse: Response in OpenAI standard format containing:
                - data: List of embedding items with vectors, indices and object type
                - model: Name of the model used
                - object: Always "list"
                - usage: Token usage information
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health check for a remote or local model"""
        pass