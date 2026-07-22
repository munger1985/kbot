from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import TypeVar, Generic
from platform_core.contracts import EmbeddingDataItem


class EmbeddingConfig(BaseModel):
    """Embedding model configuration"""
    model_name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Service provider")
    max_tokens: int = Field(..., description="Maximum number of tokens")
    batch_size: int = Field(..., description="Batch processing size")

T = TypeVar("T", bound=EmbeddingConfig)

class EmbeddingResponse(BaseModel):
    """Embedding response"""
    data: list[EmbeddingDataItem] = Field(..., description="List of embedding data items")
    model: str = Field(..., description="Name of the embedding model used")
    object: str = Field("list", description="Object type (always 'list')")
    usage: dict[str, int] = Field(..., description="Token usage information")


class BaseEmbedding(ABC, Generic[T]):
    """
    Abstract base class for embedding models.
    Defines standard interfaces to be implemented by all embedding model classes.
    """
    def __init__(self, config: T) -> None:
        """Initialize embedding model with configuration
        
        Args:
            config: Embedding model configuration object
        """
        self.config: T = config
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
        self.batch_size = config.batch_size

    @abstractmethod
    async def startup(self) -> None:
        """
        Initialize embedding model and create client connections.
        
        Raises:
            RuntimeError: Raised when initialization fails
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Close embedding model and client connections.
        
        Raises:
            RuntimeError: Raised when errors occur during shutdown process
        """
        pass
    
    def _build_standard_response(
        self, 
        embeddings: list[list[float]], 
        model_name: str, 
        tokens: int = 0
    ) -> EmbeddingResponse:
        """
        Unified method to build OpenAI-standard compliant response object.
        
        Args:
            embeddings: Nested list of embedding vectors
            model_name: Identifier of the model used
            tokens: Total number of tokens consumed
        """
        data = [
            EmbeddingDataItem(
                embedding=emb,
                index=i,
                object="embedding"
            ) for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            data=data,
            model=model_name,
            object="list",
            usage={
                "prompt_tokens": tokens,
                "total_tokens": tokens
            }
        )

    def _build_empty_response(self, model_name: str) -> EmbeddingResponse:
        """Unified method to build empty response object"""
        return EmbeddingResponse(
            data=[],
            model=model_name,
            object="list",
            usage={"prompt_tokens": 0, "total_tokens": 0}
        )

    @abstractmethod
    async def embed(self, texts: list[str], batch_size: int | None = None, is_query: bool = True) -> EmbeddingResponse:
        """
        Generate embedding vectors for list of texts, following OpenAI standard format.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Batch processing size (uses default if None)
            is_query: Whether the texts are query inputs (default: True)
            
        Returns:
            EmbeddingResponse: OpenAI-standard compliant response object containing:
                - data: List of embedding data items (vector, index, object type)
                - model: Name of the model used
                - object: Always "list"
                - usage: Token usage statistics
                
        Raises:
            ValueError: Raised when input texts are empty or invalid
            RuntimeError: Raised when model is uninitialized or errors occur during processing
            RateLimitError: Raised when rate limits are exceeded
            APIError: Raised when API calls fail
        """
        pass
