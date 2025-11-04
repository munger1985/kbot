from loguru import logger
from utils.call_models import CallModel
from .agent_params import KBResult, AgentParams


class AgentRerank:
    """智能体重排类"""
    
    def __init__(self, agent_params: AgentParams):
        """
        初始化智能体重排器
        
        Args:
            agent_params: 智能体参数配置
        """
        self.agent_params = agent_params

    async def rerank_kb(self, question: str, kb_results: list[KBResult]) -> list[KBResult]:
        """
        对知识库结果进行重排
        
        Args:
            question: 查询文本
            kb_results: 需要重排的KBResult对象列表
            
        Returns:
            list[KBResult] | None: 更新了reranker_scores的重排后KBResult列表，出错时返回None
        """
        if not kb_results:
            logger.warning("未提供知识库结果进行重排")
            return kb_results
        
        if self.agent_params.reranker_model_id is None:
            logger.warning("未配置重排模型，跳过重排")
            return kb_results
            
        # 从KBResult对象中提取内容
        contents = [result.content for result in kb_results]
        
        # 获取重排参数
        top_k = getattr(self.agent_params, 'reranker_top_k', None)
        
        # 调用重排模型
        rerankers = await CallModel().call_reranker_model(
            model_id=self.agent_params.reranker_model_id,
            query=question,
            documents=contents,
            top_k=top_k
        )
        
        if rerankers is None:
            logger.warning("重排模型调用失败或返回为空")
            return kb_results

        reranked_results: list[KBResult] = []

        # 处理重排结果
        for reranker in rerankers:
            index = reranker.get("index")
            score = reranker.get("score")
            
            if index is not None and score is not None and 0 <= index < len(kb_results):
                # 创建新的KBResult对象，保留原始属性并更新重排分数
                original_result = kb_results[index]
                reranked_result = KBResult(
                    file_id=original_result.file_id,
                    chunk_type=original_result.chunk_type,
                    page_num=original_result.page_num,
                    content=original_result.content,
                    similarity=original_result.similarity,
                    weight=original_result.weight,
                    reranker_score=score
                )
                reranked_results.append(reranked_result)

                logger.debug(f"知识库结果内容片段: {reranked_result.content[:20]}...")
                logger.debug(f"知识库结果重排得分(归一化后): {reranked_result.reranker_score}")
                logger.debug(f"知识库结果权重: {reranked_result.weight}")
            else:
                logger.warning(f"无效的重排结果索引或分数: index={index}, score={score}")

        logger.debug(f"使用重排模型 {self.agent_params.reranker_model_id} 重排了 {len(reranked_results)} 个结果")

        return reranked_results