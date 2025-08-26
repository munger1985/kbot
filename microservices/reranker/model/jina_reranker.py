
import torch
from loguru import logger
from pydantic import Field
from .local_reranker import LocalReranker, LocalRerankerConfig


class JinaRerankerConfig(LocalRerankerConfig):
    """Jina 专用的配置"""
    model_name: str = Field("jinaai/jina-reranker-v2-base", description="Jina model name")
    max_tokens: int = Field(512, description="Jina models typically use 512 tokens")
    use_fp16: bool = Field(True, description="Jina models benefit from FP16")

    
class JinaReranker(LocalReranker):
    """专门适配 Jina Reranker 的类"""
    
    async def rerank(self, query: str, documents: list[str], top_k: int | None = None):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Call startup() first.")
        
        if not documents:
            return []
        
        top_k = min(top_k, len(documents)) if top_k else len(documents)
        
        try:
            # 为 Jina 准备特定的输入格式
            pairs = []
            for doc in documents:
                # Jina 的标准输入格式
                jina_input = f"Query: {query} Document: {doc}"
                pairs.append(jina_input)
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_tokens
                )
                
                if self.device_map is None:
                    inputs = inputs.to(self.device)
                
                # Jina 分数处理
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()
            
            
            # Create list of (index, score) tuples
            scored_results = [(i, score) for i, score in enumerate(scores)]
            
            # Sort by score in descending order
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Limit to top_k results
            scored_results = scored_results[:top_k]
            
            # Return results in requested format
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.exception(f"Error during reranking: {str(e)}")
            raise