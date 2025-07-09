import numpy as np
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Optional
from prometheus_client import Counter, Histogram


class BaseEmbeddingConfig(BaseModel):
    model_name: str
    dimension: int
    max_tokens: int = 512
    
class LocalEmbeddingConfig(BaseEmbeddingConfig):
    model_path: str
    device: str = "cuda:0"
    trust_remote_code: bool = False
    
class CloudEmbeddingConfig(BaseEmbeddingConfig):
    provider: str
    api_key: str
    endpoint: Optional[str] = None
    timeout: int = 30

class BaseEmbedding(ABC):
    ERROR_COUNTER = Counter(
        'embedding_errors_total', 
        'Total number of embedding errors', 
        ['provider']
    )
    
    LATENCY_HIST = Histogram(
        'embedding_latency_seconds', 
        'Embedding latency in seconds',
        ['model_type']
    )
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize the embedding model"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the embedding model"""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        pass