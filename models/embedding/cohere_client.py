from typing import List, Optional, Any
import numpy as np
import cohere
from cohere import AsyncClient
from prometheus_client import Histogram, Counter
from models.embedding.base import BaseEmbedding, RemoteEmbeddingConfig
from core.config import settings


class CohereEmbedding(BaseEmbedding):
    """
    Production-ready Cohere embedding client with advanced features.
    
    Example:
        >>> embedder = CohereEmbedding(
        ...     api_key="your-cohere-key",
        ...     model_name="embed-english-v3.0",
        ...     timeout=30,
        ...     input_type="search_document"
        ... )
        >>> await embedder.startup()
        >>> embeddings = await embedder.embed(
        ...     texts=["Hello world"],
        ...     batch_size=96,  # Cohere's optimal batch size
        ...     truncate="END"  # Cohere-specific param
        ... )
        >>> await embedder.shutdown()
    """
    
    # Prometheus metrics with Cohere-specific labels
    LATENCY_HIST = Histogram(
        'cohere_embedding_latency_seconds',
        'Latency for Cohere embedding requests',
        ['model_name', 'input_type']
    )
    
    ERROR_COUNTER = Counter(
        'cohere_embedding_errors_total',
        'Count of Cohere embedding errors',
        ['model_name', 'error_code']
    )
    
    REQUEST_COUNTER = Counter(
        'cohere_embedding_requests_total',
        'Count of Cohere embedding requests',
        ['model_name']
    )

    def __init__(self, config: RemoteEmbeddingConfig):
        """
        Initialize Cohere embedding client.
        
        Args:
            api_key: Cohere API key
            model_name: Cohere model name (e.g., "embed-english-v3.0")
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            input_type: Cohere embedding type ("search_document", "search_query" etc)
            default_truncate: How to handle long texts ("END"|"START"|"NONE")
            kwargs: Additional Cohere client parameters
        """
        self._client: Optional[AsyncClient] = None
        self.api_key = config.api_key
        self.model_name = config.model_name
        self.timeout = config.timeout or settings["embed"]["timeout"]
        self._cohere_params = config.additional_params
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize async Cohere client with retry configuration."""
        if self._is_initialized:
            return
            
        self._client = cohere.AsyncClient(
            api_key=self.api_key,
            timeout=self.timeout,
            **self._cohere_params
        )
        self._is_initialized = True

    async def shutdown(self) -> None:
        """Cleanup client resources (Cohere's AsyncClient doesn't require explicit close)."""
        self._client = None
        self._is_initialized = False

    async def embed(
        self,
        texts: List[str],
        batch_size: int = 96,  # Cohere's recommended batch size
        truncate: Optional[str] = "END", # 或 "START"/"NONE"
        input_type: str = "search_document", # 或 "search_query"
        raise_on_error: bool = True,
        **embed_kwargs: Any
    ) -> np.ndarray:
        """
        Generate embeddings with Cohere-specific optimizations.
        
        Args:
            texts: Input texts to process
            batch_size: Texts per request (Cohere recommends 96 for v3 models)
            truncate: How to handle long texts ("END"|"START"|"NONE")
            input_type: Override default input_type
            raise_on_error: Whether to raise exceptions
            embed_kwargs: Additional Cohere embed parameters
            
        Returns:
            np.ndarray: Embedding matrix of shape (texts, embedding_dim)
            
        Raises:
            RuntimeError: If client not initialized
            cohere.CohereError: For API-specific errors
        """
        if not self._is_initialized:
            raise RuntimeError("Cohere client not initialized. Call startup() first.")
            
        if not texts:
            return np.array([])

        # Use instance defaults if not specified
        current_model = embed_kwargs.get('model', self.model_name)
        
        self.REQUEST_COUNTER.labels(model_name=current_model).inc()
        embeddings = []

        try:
            with self.LATENCY_HIST.labels(
                model_name=current_model,
                input_type=input_type
            ).time():
                # Cohere-specific batch processing
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    response = await self._client.embed( # type: ignore
                        texts=batch,
                        model=current_model,
                        input_type=input_type,
                        truncate=truncate,
                        **embed_kwargs
                    )
                    
                    embeddings.extend(response.embeddings)
                
            return np.array(embeddings) if embeddings else np.array([])
                       
        except Exception as e:
            self.ERROR_COUNTER.labels(
                model_name=current_model,
                error_code="unknown"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])

    @property
    def is_initialized(self) -> bool:
        """Check if client is ready for requests."""
        return self._is_initialized

    @property
    def embedding_dimension(self) -> int:
        """Get dimension of embeddings based on model."""
        # Cohere model dimensions mapping
        dims = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384
        }
        return dims.get(self.model_name, 1024)  # Default to 1024 for unknown models
