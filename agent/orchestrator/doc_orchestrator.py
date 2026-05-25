# services/orchestrator/doc_orchestrator.py
from loguru import logger
from typing import Any
from core.database.oracle import get_session

from services.search.doc_service import DocService
from services.kb import ModelParams
from core.exceptions import *
from dao.entities import AgentEntity
from dao.repositories import AgentRepository


class DocOrchestrator:
    def __init__(self):
        self.doc_service = DocService()

    @property
    def db_session(self):
        return get_session()

    async def run_pipeline(
        self,
        agent_id: int,
        standalone_query: str,  # 接收 Root Agent 改写后的问题
        search_keywords: str,    # 接收 Root Agent 提取的关键词
        security_level: int,
        tags: list[str] = []
    ) -> dict[str, Any]:
        """
        专注于检索流程的编排。
        注意：此处不再处理记忆回写和 Prompt 组装，只负责提供“素材”。
        """
        async with self.db_session as session:
            # 1. 获取 Agent 和模型配置 (复用原有的 _get_agent_and_params 逻辑)
            agent_repo = AgentRepository(session)
            agent = await agent_repo.get_by_id(agent_id)
            model_params = await self._get_model_params(agent) 

            # 2. 执行完整的检索流水线 (调用 DocService)
            kb_results, query_vec = await self.doc_service.get_knowledge_context(
                db_session=session,
                agent_id=agent_id,
                question=standalone_query,
                keywords=search_keywords,
                security_level=security_level,
                model_params=model_params,
                tags=tags
            )

        # 3. 返回检索结果及其元数据
        # 供后续 MultiSkillOrchestrator 放入 context_memory 或进行下一步推理
        return {
            "kb_results": kb_results,
            "query_vec": query_vec,
            "model_params": model_params,
            "agent_prompt": agent.prompt_id # 仅返回引用，不在此处查询具体内容
        }

    async def _get_model_params(self, agent: AgentEntity) -> ModelParams:
        """
        从ID解析模型名称，并组织模型参数

        Args:
            agent: 智能体实体

        Returns:
            模型参数对象
        """
        if not agent.models:
            logger.error(f"智能体 {agent.agent_id} 未配置任何模型")
            raise NotFoundError(f"智能体 {agent.agent_id} 未配置模型")
        
        llm_model = agent.models.get("llm_model")
        emb_model = agent.models.get("txt_embedding_model")
        rerank_model = agent.models.get("rerank_model")
        if not llm_model or not emb_model:
            logger.error(f"智能体 {agent.agent_id} 未配置有效大模型/嵌入模型")
            raise NotFoundError("智能体未配置有效大模型或嵌入模型")

        llm_params = {
            "top_k": agent.models.get("llm_top_k"),
            "top_p": agent.models.get("llm_top_p"),
            "temperature": agent.models.get("llm_temperature"),
            "max_tokens": agent.models.get("llm_max_tokens")
        }
        # 移除空值参数
        for k, v in list(llm_params.items()):
            if not v:
                llm_params.pop(k)

        logger.debug(f"大模型参数：{llm_params}")

        params = ModelParams(
            llm_model=llm_model,
            llm_params=llm_params,
            txt_embedding_model=emb_model,
            img_embedding_model="",
            vlm_model="",
            do_rerank=agent.models.get("do_rerank", False),
            rerank_model= rerank_model or "",
            rerank_top_k=agent.models.get("rerank_top_k", 10)
        )
        return params