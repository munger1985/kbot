import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
from .base import LocalEmbeddingConfig, BaseEmbedding

class LocalEmbedding(BaseEmbedding):
    def __init__(self, config: LocalEmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.device
        
    async def startup(self) -> None:
        """Initialize the embedding model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.model = AutoModel.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            if torch.cuda.is_available() and "cuda" in self.device:
                self.model = self.model.to(self.device)
                
        except Exception as e:
            self.ERROR_COUNTER.labels(provider="local").inc()
            raise
    
    async def shutdown(self) -> None:
        """Release resources"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache()
    
    @lru_cache(maxsize=1000)
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        with self.LATENCY_HIST.labels(model_type='local').time():
            try:
                # Tokenize the texts
                encoded_input = self.tokenizer( # type: ignore
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_tokens,
                    return_tensors='pt'
                )
                
                # Move to device
                if torch.cuda.is_available() and "cuda" in self.device:
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input) # type: ignore
                    
                # Mean pooling - take attention mask into account for averaging
                attention_mask = encoded_input['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(model_output.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy array
                return embeddings.cpu().numpy()
                
            except Exception as e:
                self.ERROR_COUNTER.labels(provider="local").inc()
                raise