import torch
from loguru import logger
from pydantic import Field
from .local_reranker import LocalReranker, LocalRerankerConfig


class JinaRerankerConfig(LocalRerankerConfig):
    """Jina Reranker 专用配置"""
    model_name: str = Field("jinaai/jina-reranker-v2-base", description="Jina 模型名称")
    max_tokens: int = Field(512, description="Jina 模型通常使用 512 tokens")
    use_fp16: bool = Field(True, description="Jina 模型受益于 FP16")

    
class JinaReranker(LocalReranker):
    """专门适配 Jina Reranker 的类"""
    
    async def rerank(self, query: str, documents: list[str], top_k: int | None = None):
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 需要重排序的文档列表
            top_k: 返回的顶部文档数量（None 表示返回所有）
            
        Returns:
            包含 'index' 和 'score' 键的字典列表
            
        Raises:
            RuntimeError: 模型未初始化时抛出
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
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
                logits = self.model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(logits).cpu().tolist()
                
            # 创建 (索引, 分数) 元组列表
            scored_results = [(i, score) for i, score in enumerate(scores)]
            
            # 按分数降序排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 限制到 top_k 个结果
            scored_results = scored_results[:top_k]
            
            # 清理内存
            del inputs, logits, scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 返回请求格式的结果
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.exception(f"重排序过程中发生错误: {str(e)}")
            raise