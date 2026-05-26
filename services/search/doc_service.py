# services/doc_service.py
import time
import asyncio
from loguru import logger

from dao.repositories import AgentConfRepository
from services.search.kb_search import TxtBaseSearchResult, TxtBaseSearch
from services.search.rerank import TxtBaseRerank
from utils.clients import AIModelClient
from core.exceptions import ParamValueError, InternalServerError
from services.kb import ModelParams

class DocService:
    def __init__(self):
        self.tb_search = TxtBaseSearch()
        self.rerank_client = TxtBaseRerank()
        self.model_client = AIModelClient()

    async def get_knowledge_context(
        self,
        db_session,
        agent_id: int,
        question: str,
        keywords: str,
        security_level: int,
        model_params: ModelParams,  # 传入解析好的 ModelParams 对象
        tags: list[str] = []
    ) -> tuple[list[TxtBaseSearchResult], list[float]]:
        """
        核心业务流水线：向量化 -> 并行检索 -> 重排序
        """
        # 1. 向量化 (验证问题有效性)
        query_vec = await self._get_embedding(question, model_params.txt_embedding_model, agent_id)

        # 2. 获取知识库配置
        conf_repo = AgentConfRepository(db_session)
        agent_confs = await conf_repo.get_by_agent(agent_id)

        # 3. 并行检索
        logger.info(f"开始为智能体 {agent_id} 执行知识库检索，安全等级：{security_level}")
        start_time = time.time()
        
        # 组装任务池
        tb_tasks = []
        for conf in agent_confs:
            tb_tasks.append(self.tb_search.search(
                int(conf.kb_id),
                str(keywords),
                int(conf.search_top_k or 10),
                float(conf.search_score_threshold or 0.0),
                conf.do_rerank,
                float(conf.tool_weight or 1.0),
                int(security_level),
                query_vec,
                tags
            ))

        # 执行并行任务
        raw_results = await asyncio.gather(*tb_tasks, return_exceptions=True)
        retrieved_results = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.error(f"知识库任务 {i} 执行失败：{res}")
                continue
            elif isinstance(res, list):
                retrieved_results.extend(res)
                logger.info(f"知识库任务 {i} 成功返回 {len(res)} 条结果")
            else:
                logger.warning(f"知识库任务 {i} 返回结果为空或格式错误")

        # 4. 重排序逻辑
        final_results = await self._apply_rerank(
            retrieved_results, question, model_params
        )

        logger.info(f"知识库检索完成，耗时：{time.time() - start_time:.2f}s，最终返回 {len(final_results)} 条结果")
        return final_results, query_vec

    async def _get_embedding(self, question: str, model_name: str, agent_id: int) -> list[float]:
        question = question.strip() if question else ""
        if not question:
            raise ParamValueError(f"智能体 {agent_id} 的检索问题不能为空")
        
        vec = await self.model_client.get_embedding(model_name, question)
        if not vec:
            raise InternalServerError("嵌入向量生成失败")
        return vec

    async def _apply_rerank(self, results: list[TxtBaseSearchResult], question: str, params: ModelParams) -> list[TxtBaseSearchResult]:
        if not params.do_rerank or not params.rerank_model or len(results) <= 1:
            logger.debug(f"检索结果数量不足或未配置重排序模型，直接返回原始结果")
            return results
        
        return await self.rerank_client.rerank(
            model_name=params.rerank_model,
            top_k=int(params.rerank_top_k or 10),
            question=question,
            kb_results=results
        )