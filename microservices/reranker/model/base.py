"""
Base configuration and interface definitions for Reranker models.

This module contains:
1. RerankerConfig: Base configuration class for reranker models
2. BaseReranker: Abstract base class defining the core reranker interface
"""

from typing import Any, TypeVar, Generic
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class RerankerConfig(BaseModel):
    """Base configuration class for reranker models.
    
    Contains core configuration parameters required to initialize and interact
    with any reranker model implementation, including model identification
    and input constraints.
    """
    model_name: str = Field(..., description="Name of the reranker model (e.g., 'bge-reranker-large')")
    provider: str = Field(..., description="Provider/vendor of the reranker model (e.g., 'local', 'cohere', 'jina')")
    max_tokens: int = Field(4096, description="Maximum input sequence length in tokens")

# Generic type variable bound to RerankerConfig for type-safe configuration inheritance
T = TypeVar("T", bound=RerankerConfig)

class BaseReranker(ABC, Generic[T]):
    """Abstract base class for all reranker model implementations.
    
    Defines the core interface contract for reranker models, including
    lifecycle management (startup/shutdown) and the core reranking functionality.
    All concrete reranker implementations must inherit from this class and implement
    all abstract methods.
    
    Type Parameters:
        T: A subclass of RerankerConfig containing provider-specific configuration
    """

    def __init__(self, config: T) -> None:
        """Initialize reranker instance with configuration.
        
        Args:
            config: Reranker configuration object containing model-specific settings
        """
        self.config: T = config
        self.provider = config.provider
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
    
    @abstractmethod
    async def startup(self) -> None:
        """Asynchronously initialize reranker model resources.
        
        This method should handle any provider-specific initialization such as:
        - Loading model weights into memory
        - Creating API clients for remote reranker services
        - Initializing tokenizers
        - Setting up connection pools
        
        Called once during service initialization before any reranking requests.
        """
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Reorder documents by relevance to the given query.
        
        Core method for reranking a list of documents based on their semantic
        relevance to the input query. Returns results sorted from most relevant
        to least relevant, with optional top-k filtering.
        
        Args:
            query: Input query text to measure relevance against
            documents: List of document texts to be reranked
            top_k: Number of top relevant documents to return (None returns all documents)
            
        Returns:
            list[dict[str, Any]]: List of reranked results, each containing at minimum:
                - "text": The original document text
                - "relevance_score": Numerical relevance score (higher = more relevant)
                - "index": Original position in the input documents list
                
            The list is sorted in descending order of relevance score.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Asynchronously clean up reranker resources.
        
        This method should handle cleanup operations such as:
        - Unloading model weights from memory
        - Closing API connections
        - Releasing GPU memory
        - Cleaning up temporary resources
        
        Called once during service shutdown to ensure graceful termination.
        """
        pass