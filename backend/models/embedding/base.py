import numpy as np
import torch
from abc import ABC, abstractmethod
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from typing import List, Optional


class EmbeddingConfig(BaseModel):
    model_name: str
    provider: str
    max_tokens: int

class LocalEmbeddingConfig(EmbeddingConfig):
    model_path: Optional[str] = None
    device: Optional[str] = None
    device_map: Optional[str] = None
    max_memory: Optional[int] = None
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
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        pass