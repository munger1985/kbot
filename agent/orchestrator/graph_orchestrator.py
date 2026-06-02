# agent/orchestrator/graph_orchestrator.py
from typing import Any
from loguru import logger

from core.database.oracle import get_session
from core.exceptions import NotFoundError
from dao.repositories import AgentRepository
from services.graph import GraphService
from services.kb import ModelParams

class GraphOrchestrator:
    def __init__(self):
        self.graph_service = GraphService()

    @property
    def db_session(self):
        return get_session()

    async def run_pipeline(
        self,
        agent_id: int,
        standalone_query: str,
        vertex_names: list[str],
        security_level: int,
        tags: list[str] = []
    ) -> dict[str, Any]:
        """
        专注于图谱检索流程的管道编排，不污染记忆状态，纯净返回拓扑素材
        """
        async with self.db_session as session:
            # 1. 获取 Agent 和多模态大模型配置
            agent_repo = AgentRepository(session)
            agent = await agent_repo.get_by_id(agent_id)
            if not agent:
                raise NotFoundError(f"未能查询到智能体元数据：{agent_id}")
                
            model_params = await self._get_model_params(agent)

            # 2. 执行完整的图检索管道
            graph_results, query_vec = await self.graph_service.get_graph_context(
                db_session=session,
                agent_id=agent_id,
                question=standalone_query,
                vertex_names=vertex_names,
                security_level=security_level,
                model_params=model_params,
                tags=tags
            )

        return {
            "graph_results": graph_results,
            "query_vec": query_vec,
            "model_params": model_params,
            "agent_prompt": agent.prompt_id
        }

    async def _get_model_params(self, agent) -> ModelParams:
        """从 Agent 的结构化实体字典中提取统一的模型驱动参数对象"""
        if not agent.models:
            raise NotFoundError(f"智能体 {agent.agent_id} 未配置任何模型结构")
        
        llm_model = agent.models.get("llm_model")
        emb_model = agent.models.get("txt_embedding_model")
        if not llm_model or not emb_model:
            raise NotFoundError("智能体未配置有效大模型或嵌入模型")

        llm_params = {k: v for k, v in {
            "top_k": agent.models.get("llm_top_k"),
            "top_p": agent.models.get("llm_top_p"),
            "temperature": agent.models.get("llm_temperature"),
            "max_tokens": agent.models.get("llm_max_tokens")
        }.items() if v}

        return ModelParams(
            llm_model=llm_model,
            llm_params=llm_params,
            txt_embedding_model=emb_model,
            img_embedding_model="",
            vlm_model="",
            do_rerank=False,  # 图节点暂时不需要标准 TxtRerank，由算法算距离
            rerank_model="",
            rerank_top_k=0
        )