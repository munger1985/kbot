# agent/agent/graph_agent.py
import uuid
from loguru import logger
from typing import Any

from platform_core.database.oracle import get_session
from dao.repositories import GraphRepository
from services.search import TxtBaseSearchResult
from agent.memory import MemoryService
from agent.orchestrator.graph_orchestrator import GraphOrchestrator

class GraphAgent:
    """图谱智能体核心类，提供图查询、拓扑富化及图路径回填功能"""

    def __init__(self):
        self.memory_service = MemoryService()
        self.orchestrator = GraphOrchestrator()

    @property
    def db_session(self):
        return get_session()

    async def graph_retrieval(
        self,
        session_id: str,
        agent_id: int,
        question: str,
        standalone_query: str,
        vertex_names: list[str],
        security_level: int,
        user_id: str,
        tags: list[str] = []
    ) -> list[dict]:
        """
        图谱核心检索入口，保障会话总线连续性
        """
        # 1. 确保会话生命周期完备
        await self.memory_service.ensure_session_exists(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            question=question
        )

        # 2. 调起流程管道获取拓扑关系素材
        pipe_out = await self.orchestrator.run_pipeline(
            agent_id=agent_id,
            standalone_query=standalone_query,
            vertex_names=vertex_names,
            security_level=security_level,
            tags=tags
        )

        # 3. 将检索出来的图谱碎片融合知识库和业务元数据（例如实体属性、别名等）
        enriched_graph = await self._enrich_graph_with_metadata(pipe_out['graph_results'])
        
        return enriched_graph

    async def _enrich_graph_with_metadata(self, graph_results: list[TxtBaseSearchResult]) -> list[dict]:
        if not graph_results:
            return []

        # 类似于 DocAgent 批量拉取唯一文件名的设计
        # 这里可以用来批量拉取图谱节点的详细标签、术语定义、或者对应的额外说明
        references = []
        for idx, res in enumerate(graph_results):
            try:
                ref = res.to_dict() if hasattr(res, 'to_dict') else dict(res)
                # 补充前端渲染所必须的默认显示字段
                if "title" not in ref:
                    ref["title"] = f"实体关联: {ref.get('source_node')} -> {ref.get('target_node')}"
                references.append(ref)
            except Exception as e:
                logger.error(f"处理图关系元数据富化异常，索引{idx}：{e}")
                
        return references