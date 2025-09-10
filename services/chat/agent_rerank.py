from loguru import logger
from .agent_params import KBResult, AgentParams
from utils.call_models import CallModel


class AgentRerank:
    """智能体重排类"""
    
    def __init__(self, agent_params: AgentParams):
        """
        初始化智能体重排器
        
        Args:
            agent_params: 智能体参数配置
        """
        self.agent_params = agent_params

    async def rerank_kb(self, question: str, kb_results: list[KBResult]) -> list[KBResult] | None:
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
        
        if self.agent_params.reranker_model_name is None:
            logger.warning("未配置重排模型，跳过重排")
            return kb_results
            
        # 从KBResult对象中提取内容
        contonts = [result.content for result in kb_results]
        
        # 调用重排模型
        rerankers = await CallModel().call_reranker_model(
            model_unique_name=self.agent_params.reranker_model_name,
            query=question,
            documents=contonts,
            top_k=self.agent_params.reranker_top_k
        )
        
        if rerankers is None:
            logger.warning("重排模型调用失败或返回为空")
            return kb_results

        reranked_results: list[KBResult] = []    
        # 更新原始KBResult对象中的reranker_score
        for reranker in rerankers:
            index = reranker.get("index")
            score = reranker.get("score")
            logger.debug(f"重排索引: {index}, 得分: {score}")

            if index is not None and score is not None:
                reranked_result = KBResult()
                reranked_result.file_id = kb_results[index].file_id
                reranked_result.chunk_type = kb_results[index].chunk_type
                reranked_result.page_num = kb_results[index].page_num
                reranked_result.content = kb_results[index].content
                reranked_result.similarity = kb_results[index].similarity
                reranked_result.weight = kb_results[index].weight
                reranked_result.reranker_score = score
                reranked_results.append(reranked_result)

                logger.debug(f"知识库结果内容片段: {reranked_result.content[0:20]}")
                logger.debug(f"知识库结果重排得分: {reranked_result.reranker_score}")
                logger.debug(f"知识库结果权重: {reranked_result.weight}")
                

        logger.debug(f"使用重排模型 {self.agent_params.reranker_model_name} 重排了 {len(reranked_results)} 个结果")

        return reranked_results