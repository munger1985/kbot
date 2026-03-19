from pydantic import BaseModel, Field
from typing import Any


# Define request models
class RerankerRequest(BaseModel):
    """Request model for document reranking operations.
    
    Specifies the reranker model to use, query text, documents to rerank,
    and the number of top relevant documents to return.
    """
    model_name: str = Field(..., description="Unique technical name of the reranker model")
    query: str = Field(..., description="Query text to measure document relevance against")
    documents: list[str] = Field(..., description="List of document texts to be reranked")
    top_k: int | None = Field(10, description="Number of top relevant documents to return (None returns all documents)")

class ToggleModelRequest(BaseModel):
    """Request model for loading/unloading reranker models from memory.
    
    Used to dynamically manage model lifecycle in the model pool without
    restarting the application.
    """
    model_name: str = Field(..., description="Unique technical name of the model")
    operation: str = Field(..., description="Operation type: 'load' to load model into memory, 'unload' to remove model from memory")

# Define response models
class RerankerResponse(BaseModel):
    """Response model for document reranking operations.
    
    Contains the list of reranked documents with relevance scores and original indices.
    """
    rerankers: list[dict[str, Any]] = Field(..., description="List of reranked documents with relevance scores, each containing 'index' (original position) and 'score' (relevance score)")