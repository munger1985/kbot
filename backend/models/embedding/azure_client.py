from typing import List, Optional, Dict, Any
import numpy as np
import openai
from openai import AsyncAzureOpenAI
from prometheus_client import Histogram, Counter
from models.embedding.base import BaseEmbedding, RemoteEmbeddingConfig
from core.config import settings


class AzureEmbedding(BaseEmbedding):
    """
    High-performance Azure OpenAI embedding client with enterprise-grade features.
    
    Example:
        >>> embedder = AzureEmbedding(
        ...     api_key="your-azure-key",
        ...     api_version="2023-05-15",
        ...     deployment_name="your-deployment",
        ...     endpoint="https://your-resource.openai.azure.com",
        ...     timeout=30
        ... )
        >>> await embedder.startup()
        >>> embeddings = await embedder.embed(["Hello world"], batch_size=50)
        >>> await embedder.shutdown()
    """
    
    # Prometheus metrics
    LATENCY_HIST = Histogram(
        'azure_embedding_latency_seconds',
        'Latency for Azure OpenAI embedding requests',
        ['deployment', 'api_version']
    )
    
    ERROR_COUNTER = Counter(
        'azure_embedding_errors_total',
        'Count of Azure embedding errors',
        ['deployment', 'error_type']
    )
    
    REQUEST_COUNTER = Counter(
        'azure_embedding_requests_total',
        'Count of Azure embedding requests',
        ['deployment']
    )

    def __init__(self, config: RemoteEmbeddingConfig):
        """
        Initialize Azure OpenAI embedding client.
        
        Args:
            api_key: Azure OpenAI API key
            deployment_name: Deployment name (not model name)
            endpoint: Azure endpoint URL (e.g., "https://xxx.openai.azure.com")
            api_version: Azure API version (e.g., "2023-05-15")
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            custom_headers: Custom HTTP headers to include
            kwargs: Additional Azure-specific parameters
        """
        self._client: Optional[AsyncAzureOpenAI] = None
        self.api_key = config.api_key
        self.deployment_name = config.deployment_name
        self.endpoint = config.endpoint
        self.api_version = config.api_version
        self.timeout = config.timeout or settings["embed"]["timeout"]
        self.max_retries = config.max_retries or settings["embed"]["max_retries"]
        self.custom_headers = {}
        self._is_initialized = False
        self._azure_params = config.additional_params  # Store additional Azure-specific params

    async def startup(self) -> None:
        """Initialize the async Azure client with proper cleanup."""
        if self._is_initialized:
            return
            
        if not self.endpoint:
            raise ValueError("Azure endpoint must be provided")

        headers = {
            "User-Agent": "KBOT/3.0.0",
            "X-Request-Source": "backend-service",
            **self.custom_headers
        }

        self._client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=headers,
            **self._azure_params
        )
        self._is_initialized = True

    async def shutdown(self) -> None:
        """Properly cleanup client resources."""
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
        Generate embeddings with Azure-specific optimizations.
        
        Args:
            texts: Input texts to process
            batch_size: Texts per request (Azure recommends <= 16 for long texts)
            raise_on_error: Whether to raise exceptions on failure
            
        Returns:
            np.ndarray: Embedding matrix (texts x dimensions)
            
        Raises:
            RuntimeError: If client not initialized
            openai.APIError: For Azure-specific errors
        """
        if not self._is_initialized:
            raise RuntimeError("Azure client not initialized. Call startup() first.")
            
        if not texts:
            return np.array([])

        embeddings = []
        self.REQUEST_COUNTER.labels(deployment=self.deployment_name).inc()

        try:
            with self.LATENCY_HIST.labels(
                deployment=self.deployment_name,
                api_version=self.api_version
            ).time():
                # Azure-specific batch processing
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    
                    response = await self._client.embeddings.create( # type: ignore
                        model=self.deployment_name,  # Azure uses deployment names
                        input=batch,
                        encoding_format="float"  # Explicit format for numpy
                    )
                    
                    embeddings.extend([item.embedding for item in response.data])
                
            return np.vstack(embeddings) if embeddings else np.array([])
            
        except openai.RateLimitError as e:
            self.ERROR_COUNTER.labels(
                deployment=self.deployment_name,
                error_type="rate_limit"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except openai.APIConnectionError as e:
            self.ERROR_COUNTER.labels(
                deployment=self.deployment_name,
                error_type="connection"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except openai.APIStatusError as e:
            self.ERROR_COUNTER.labels(
                deployment=self.deployment_name,
                error_type=f"http_{e.status_code}"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])
            
        except Exception as e:
            self.ERROR_COUNTER.labels(
                deployment=self.deployment_name,
                error_type="unknown"
            ).inc()
            if raise_on_error:
                raise
            return np.array([])

    @property
    def is_initialized(self) -> bool:
        """Check if client is ready for requests."""
        return self._is_initialized

    @property
    def azure_config(self) -> Dict[str, Any]:
        """Get current Azure configuration."""
        return {
            "api_version": self.api_version,
            "endpoint": self.endpoint,
            "deployment": self.deployment_name,
            **self._azure_params
        }