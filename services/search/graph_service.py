# services/graph_service.py
import time
import asyncio
from loguru import logger
from typing import Any

from dao.repositories import AgentConfRepository
from services.search.graph_search import GraphBaseSearch
from services.search.kb_search import TxtBaseSearchResult
from utils.clients import AIModelClient
from core.exceptions import ParamValueError, InternalServerError
from services.kb import ModelParams

class GraphService:
    def __init__(self):
        # 假设底层图检索实现类为 GraphBaseSearch
        self.graph_search = GraphBaseSearch()
        self.model_client = AIModelClient()

    async def get_graph_context(
        self,
        db_session,
        agent_id: int,
        question: str,
        vertex_names: list[str],
        security_level: int,
        model_params: ModelParams,
        tags: list[str] = []
    ) -> tuple[list[TxtBaseSearchResult], list[float]]:
        """
        核心业务流水线：问题向量化 -> 并行图谱子图游走召回 -> 结果剪枝聚合
        """
        # 1. 向量化（如有需要，某些图谱混合检索可能用到查询向量，对齐 DocService）
        query_vec = await self._get_embedding(question, model_params.txt_embedding_model, agent_id)

        # 2. 获取该智能体挂载的知识库配置（支持 1个 Agent 挂载多个图谱知识库）
        conf_repo = AgentConfRepository(db_session)
        agent_confs = await conf_repo.get_by_agent(agent_id)

        # 3. 多路并行图谱检索
        logger.info(f"开始为智能体 {agent_id} 执行图谱空间网络检索，安全等级：{security_level}")
        start_time = time.time()
        
        graph_tasks = []
        for conf in agent_confs:
            # 优先从配置表读取图谱专属参数，若无则常数兜底
            search_top_k = int(conf.search_top_k or 5)
            # 假设图深度存储在扩展字段中，或默认取 2 
            max_depth = int(getattr(conf, "max_depth", 2) or 2)
            tool_weight = float(conf.tool_weight or 1.2)

            graph_tasks.append(self.graph_search.search_by_graph(
                kb_id=int(conf.kb_id),
                vertex_names=vertex_names,
                search_top_k=search_top_k,
                max_depth=max_depth,
                weight=tool_weight,
                security_level=int(security_level)
            ))

        # 执行并行分布式拓扑网络游走
        raw_results = await asyncio.gather(*graph_tasks, return_exceptions=True)
        retrieved_results = []
        
        for i, res in enumerate(raw_results):
            current_kb = agent_confs[i].kb_id if i < len(agent_confs) else "Unknown"
            if isinstance(res, Exception):
                logger.error(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 执行失败：{res}")
                continue
            elif isinstance(res, dict):
                # search_by_graph 返回 {"graph_result": [...]}
                graph_items = res.get("graph_result", [])
                if isinstance(graph_items, list):
                    retrieved_results.extend(graph_items)
                    logger.info(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 成功返回 {len(graph_items)} 条拓扑路径记录")
                else:
                    logger.warning(f"图谱知识库任务 {i} (KB_ID: {current_kb}) graph_result 不是列表类型")
            elif isinstance(res, list):
                # 兼容旧版直接返回列表的接口
                retrieved_results.extend(res)
                logger.info(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 成功返回 {len(res)} 条拓扑路径记录")
            else:
                logger.warning(f"图谱知识库任务 {i} (KB_ID: {current_kb}) 返回格式异常: {type(res)}")

        # 4. 图谱层面的排序/剪枝逻辑（此处可根据权重、距离直接过滤，对齐重排占位）
        final_results = self._apply_graph_filter(retrieved_results)

        logger.info(f"图谱空间检索完成，耗时：{time.time() - start_time:.2f}s，最终合并 {len(final_results)} 条实体关系边")
        return final_results, query_vec

    async def _get_embedding(self, question: str, model_name: str, agent_id: int) -> list[float]:
        question = question.strip() if question else ""
        if not question:
            raise ParamValueError(f"智能体 {agent_id} 的图谱相关检索问题不能为空")
        vec = await self.model_client.get_embedding(model_name, question)
        if not vec:
            raise InternalServerError("图谱检索前置嵌入向量生成失败")
        return vec

    def _apply_graph_filter(self, results: list[TxtBaseSearchResult]) -> list[TxtBaseSearchResult]:
        # 根据图节点或边的权值进行基本排序，确保下游 Reasoning 层能拿到关联度最高的实体属性
        results.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
        return results