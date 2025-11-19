from typing import Any
from pydantic import Field
from loguru import logger

# 优雅降级导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("警告: PyTorch 不可用，将使用备用方案")

from .local_reranker import LocalReranker, LocalRerankerConfig


class JinaRerankerConfig(LocalRerankerConfig):
    """Jina Reranker 专用配置"""
    model_name: str = Field("jinaai/jina-reranker-v2-base", description="Jina 模型名称")
    max_tokens: int = Field(512, description="Jina 模型通常使用 512 tokens")
    use_fp16: bool = Field(True, description="Jina 模型受益于 FP16")

    
class JinaReranker(LocalReranker):
    """专门适配 Jina Reranker 的类，支持优雅降级"""
    
    def _format_jina_input(self, query: str, document: str) -> str:
        """格式化 Jina Reranker 的输入"""
        return f"Query: {query} Document: {document}"
    
    def _compute_jina_scores_fallback(self, query: str, documents: list[str]) -> list[float]:
        """降级模式下的 Jina 分数计算"""
        scores = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            doc_words = set(doc.lower().split())
            
            # 计算 Jaccard 相似度
            if len(query_words) == 0 or len(doc_words) == 0:
                scores.append(0.0)
                continue
                
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Jina 特定的启发式规则
            # 1. 查询词在文档开头出现的权重更高
            first_100_chars = doc[:100].lower()
            position_bonus = 0.0
            for word in query_words:
                if word in first_100_chars:
                    position_bonus += 0.1
            
            # 2. 文档长度惩罚（Jina 偏好中等长度文档）
            doc_length = len(doc)
            if doc_length < 50:
                length_penalty = 0.3
            elif doc_length > 2000:
                length_penalty = 0.5
            else:
                length_penalty = 1.0
            
            score = (jaccard_similarity * 0.6 + position_bonus * 0.4) * length_penalty
            scores.append(min(score, 1.0))  # 确保分数在 [0, 1] 范围内
        
        return scores

    async def _process_jina_batch(self, query: str, batch_documents: list[str]) -> list[float]:
        """处理 Jina Reranker 的批次"""
        # 如果处于降级模式，使用备用方案
        if self.is_fallback_mode or not TORCH_AVAILABLE:
            return self._compute_jina_scores_fallback(query, batch_documents)
            
        if not self.model or not self.tokenizer:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        # 为 Jina 准备特定的输入格式
        pairs = []
        for doc in batch_documents:
            jina_input = self._format_jina_input(query, doc)
            pairs.append(jina_input)
        
        # 使用 inference_mode 替代 no_grad
        inference_mode = torch.inference_mode if TORCH_AVAILABLE else (lambda: lambda f: f)()
        
        with inference_mode():
            try:
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_tokens
                )
                
                if self.device_map is None:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Jina 分数处理
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
                scores = torch.sigmoid(logits).cpu().tolist()
                
                # 清理内存
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return scores if isinstance(scores, list) else [scores]
                
            except Exception as e:
                logger.error(f"Jina 批次处理失败: {e}，使用降级模式")
                return self._compute_jina_scores_fallback(query, batch_documents)
    
    async def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
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
        if not self._is_initialized:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        if not documents:
            return []
        
        top_k = min(top_k, len(documents)) if top_k else len(documents)
        
        try:
            all_scores = []
            total_docs = len(documents)
            
            if self.is_fallback_mode:
                logger.warning(f"使用降级模式对 {total_docs} 个文档进行 Jina 重排序")
                all_scores = self._compute_jina_scores_fallback(query, documents)
            else:
                # 分批处理文档
                for i in range(0, len(documents), self.batch_size):
                    batch_docs = documents[i:i + self.batch_size]
                    batch_scores = await self._process_jina_batch(query, batch_docs)
                    all_scores.extend(batch_scores)
                    
                    # 记录进度
                    if (i + len(batch_docs)) % 10 == 0 or (i + len(batch_docs)) == total_docs:
                        logger.debug(f"Jina 重排序进度: {i + len(batch_docs)}/{total_docs}")
            
            # 创建 (索引, 分数) 元组列表
            scored_results = [(i, score) for i, score in enumerate(all_scores)]
            
            # 按分数降序排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 限制到 top_k 个结果
            scored_results = scored_results[:top_k]
            
            mode_info = "降级模式" if self.is_fallback_mode else "正常模式"
            logger.info(f"Jina 重排序完成({mode_info})，返回前 {top_k} 个结果")
            
            # 返回请求格式的结果
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.exception(f"Jina 重排序过程中发生错误: {str(e)}")
            # 在错误时返回基于索引的默认排序
            return [{"index": i, "score": 1.0 - (i * 0.01)} for i in range(min(top_k, len(documents)))]

    @property
    def is_jina_model(self) -> bool:
        """检查是否为 Jina 模型"""
        return "jina" in self.model_name.lower()

    @property
    def recommended_batch_size(self) -> int:
        """获取推荐的批次大小"""
        if self.is_fallback_mode:
            return 32  # 降级模式下可以使用更大的批次
        
        if not TORCH_AVAILABLE:
            return 32
            
        # 根据设备调整批次大小
        if hasattr(self, 'device') and self.device.startswith('cuda'):
            if torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3:  # 8GB+
                return 16
            else:
                return 8
        else:
            return 4  # CPU 模式下使用较小的批次