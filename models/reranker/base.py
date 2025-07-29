from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class RerankerConfig(BaseModel):
    """Configuration for reranker models."""
    model_name: str = Field(..., description="Name of the reranker model")
    model_path: Optional[str] = Field(None, description="Optional local path to model files")
    device: Optional[str] = Field("cuda:0", description="Target device (e.g., 'cuda:0', 'cpu')")
    device_map: Optional[str] = Field(None, description="For multi-GPU setups (e.g., 'auto', 'balanced')")
    max_tokens: Optional[int] = Field(512, description="Maximum input sequence length")
    compile_model: bool = Field(True, description="Whether to compile model with torch.compile() (PyTorch 2.0+)")
    use_fp16: bool = Field(False, description="Use half-precision inference (recommended for GPU)")
    local_files_only: bool = Field(False, description="Only use local model files (no internet download)")
    trust_remote_code: bool = Field(False, description="Trust custom model code from HuggingFace")
    max_memory: Optional[Dict[str, str]] = Field(None, description="Dict of GPU memory limits (e.g. {'0': '24GB', '1': '24GB'})")


class BaseReranker(ABC):
    """Abstract base class for reranker models."""
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize the reranker model."""
        pass
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return (None for all)
            
        Returns:
            List of dicts with 'index' and 'score' keys
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up resources."""
        pass