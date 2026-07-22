from pydantic import BaseModel, Field

# Define embedding request model
class EmbeddingRequest(BaseModel):
    """Embedding request parameter model."""
    
    model_name: str = Field(..., description="Technical name of the model (e.g., gpt-4, text-embedding-ada-002)")
    texts: list[str] = Field(..., description="List of texts to generate embeddings for")
    batch_size: int | None = Field(32, description="Batch processing size")
    is_query: bool = Field(True, description="Whether the texts are query inputs")

class ToggleModelRequest(BaseModel):
    """Request model for loading/unloading models."""
    model_name: str = Field(..., description="Technical name of the model (e.g., gpt-4, text-embedding-ada-002)")
    operation: str = Field(..., description="Operation type: 'load' or 'unload'")

class SimilarityRequest(BaseModel):
    """Similarity calculation request parameter model."""
    model_name: str = Field(..., description="Technical name of the model (e.g., gpt-4, text-embedding-ada-002)")
    text1: str = Field(..., description="First text string")
    text2: str = Field(..., description="Second text string")
    method: str = Field("cosine", description="Similarity calculation method: supports 'cosine' (cosine similarity) and 'dot' (dot product)")