from typing import List, Optional
import numpy as np
import openai
from openai import AsyncOpenAI
from prometheus_client import Histogram, Counter
from models.embedding.base import BaseEmbedding, RemoteEmbeddingConfig

class OpenAIEmbedding(BaseEmbedding):
    """
    A high-performance, production-ready OpenAI embedding client with monitoring.
      
    Example:
        >>> embedder = OpenAIEmbedding(
        ...     model_name="text-embedding-3-small",
        ...     api_key="your-api-key",
        ...     timeout=30
        ... )
        >>> await embedder.startup()
        >>> embeddings = await embedder.embed(["Hello world"])
        >>> await embedder.shutdown()
    """
    
    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'openai_embedding_latency_seconds',
        'Latency for OpenAI embedding requests',
        ['model_type']
    )
    
    ERROR_COUNTER = Counter(
        'openai_embedding_errors_total',
        'Count of OpenAI embedding errors',
        ['provider', 'error_type']
    )
    
    REQUEST_COUNTER = Counter(
        'openai_embedding_requests_total',
        'Count of OpenAI embedding requests',
        ['model_type']
    )

    def __init__(self, config: RemoteEmbeddingConfig):
        """
        Initialize the OpenAI embedding client.
        
        Args:
            model_name: OpenAI embedding model name (e.g., "text-embedding-3-small")
            api_key: OpenAI API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            organization: Optional organization ID
        """
        self._client: Optional[AsyncOpenAI] = None
        self.model_name = config.model_name
        self.api_key = config.api_key
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.organization = config.organization
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize the async OpenAI client."""
        if self._is_initialized:
            return
            
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            organization=self.organization
        )
        self._is_initialized = True

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.close()
        self._client = None
        self._is_initialized = False

    async def embed(
        self,
        texts: List[str],
        batch_size: int = 100,
        raise_on_error: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process per request
            raise_on_error: Whether to raise exceptions or return empty array
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (num_texts, embedding_dim)
            
        Raises:
            RuntimeError: If client is not initialized
            openai.OpenAIError: For API-related errors
        """
        if not self._is_initialized:
            raise RuntimeError("Client not initialized. Call startup() first.")
            
        if not texts:
            return np.array([])

        embeddings = []
        self.REQUEST_COUNTER.labels(model_type=self.model_name).inc()

        try:
            with self.LATENCY_HIST.labels(model_type=self.model_name).time():
                # Process in batches if the input is large
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    response = await self._client.embeddings.create( # type: ignore
                        model=self.model_name,
                        input=batch,
                        encoding_format="float"  # Ensure float return type
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
            return np.array(embeddings)
            
        except openai.RateLimitError as e:
            self.ERROR_COUNTER.labels(
                provider="openai",
                error_type="rate_limit"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except openai.APIConnectionError as e:
            self.ERROR_COUNTER.labels(
                provider="openai",
                error_type="connection"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except openai.APIError as e:
            self.ERROR_COUNTER.labels(
                provider="openai",
                error_type="api"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except Exception as e:
            self.ERROR_COUNTER.labels(
                provider="openai",
                error_type="unknown"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self._is_initialized

    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings for the configured model."""
        # Model dimension mapping (update as needed)
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model_name)