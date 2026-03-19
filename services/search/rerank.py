from loguru import logger
from utils.clients import AIModelClient
from .result import TxtBaseSearchResult


class TxtBaseRerank:
    """知识库重排类 - 支持层级感知重排"""
    def __init__(self):
        self.model_client = AIModelClient()

    async def rerank(self, 
                     model_name: str, 
                     top_k: int, 
                     question: str, 
                     kb_results: list[TxtBaseSearchResult],
                     min_rerank_score: float = 0.01  # 新增：最小分数过滤
                    ) -> list[TxtBaseSearchResult]:
        """
        对知识库结果进行重排
        
        Args:
            model_name: 重排模型名称 (如 bge-reranker-v2-m3)
            top_k: 最终保留并返回的候选集数量
            question: 用户原始查询
            kb_results: 经过 search_by_hybrid 召回的 100 条结果
            min_rerank_score: 过滤掉低于此分数的噪音
        """ 
        if not kb_results:
            return []

        # 1. 提取内容。此时 contents 已包含 search.py 注入的 [层级路径] 前缀
        # Cross-Encoder 会利用这些标题信息来校准正文的相关度
        contents = [result.content for result in kb_results]
        
        # 2. 调用重排模型
        try:
            response = await self.model_client.call_reranker_model(
                model_name=model_name,
                query=question,
                documents=contents,
                top_k=len(kb_results) # 建议对全量召回集进行评分，后续再截断
            )
        except Exception as e:
            logger.error(f"重排模型调用异常: {e}")
            return kb_results[:top_k] # 异常回退：直接返回召回层的前 top_k

        if not response:
            logger.warning("重排模型未返回有效结果")
            return kb_results[:top_k]

        reranked_results: list[TxtBaseSearchResult] = []

        # 3. 映射结果并应用过滤
        for item in response:
            index = item.get("index")
            score = item.get("score")
            
            if index is not None and 0 <= index < len(kb_results):
                # 过滤极低相关性的碎片
                if score < min_rerank_score: # type: ignore
                    continue
                    
                target_result = kb_results[index]
                target_result.rerank_score = float(score) # type: ignore
                
                # 这种模式下，我们甚至可以微调 weight 对 rerank_score 的影响
                # 最终排序分 = 重排原生分 * 知识库业务权重
                # target_result.rerank_score *= target_result.weight 

                reranked_results.append(target_result)
            else:
                logger.warning(f"无效的重排索引: {index}")

        # 4. 根据重排分数进行最终排序
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # 打印 Top 3 结果便于观测分层效果
        for r in reranked_results[:3]:
            logger.debug(f"重排 Top 命中 | 分数: {r.rerank_score:.4f} | 内容: {r.content[:40]}...")

        logger.info(f"使用模型 {model_name} 完成重排，从 {len(kb_results)} 条中筛选出 {len(reranked_results)} 条，取 Top {top_k}")
        try:
            safe_top_k = int(top_k)
        except Exception as e:
            safe_top_k = 1
        return reranked_results[:safe_top_k]