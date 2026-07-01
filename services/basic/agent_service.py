from loguru import logger
from typing import Any
from core.database.oracle import get_session
from core.exceptions import *
from dao.repositories import (AgentRepository, AgentConfRepository,
                             PromptRepository, MemoryRepository,
                             OpsAgentConfRepository)
from services.kb.schema import ModelParams



class AgentService:
    """Agent service class for managing knowledge base search and AI chat interactions."""

    @property
    def oracle_session(self):
        """Provides a database session instance."""
        return get_session()
    
    async def feedback(self, entry_id: str, feedback: int):
        """Updates user feedback for a specific chat record."""
        async with self.oracle_session as session:
            repo = MemoryRepository(session)
            await repo.update_feedback(entry_id, feedback)
            logger.info(f"Updated feedback {feedback} for memory {entry_id}")

    async def get_context_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves history for a specific chat session."""
        async with self.oracle_session as session:
            repo = MemoryRepository(session)
            return await repo.get_sessions(session_id)
        


    async def remove_agent(self, agent_id: int, del_prompt: bool = False):
        """Removes agent, its configurations, and optionally its associated prompts."""
        try:
            async with self.oracle_session as session:
                agent_repo = AgentRepository(session)
                if del_prompt:
                    agent = await agent_repo.get_by_id(agent_id)
                    if agent and agent.prompt_id:
                        await PromptRepository(session).delete(agent.prompt_id)
                        logger.info(f"Deleted prompt {agent.prompt_id} for agent {agent_id}")

                await AgentConfRepository(session).delete_by_agent_id(agent_id)
                await OpsAgentConfRepository(session).delete_by_agent_id(agent_id)
                await agent_repo.delete(agent_id)
                
                sess_repo = MemoryRepository(session)
                await sess_repo.remove_context_by_agent(agent_id)
                logger.info(f"Successfully removed agent {agent_id} and related sessions.")
        except Exception as e:
            logger.error(f"Error removing agent {agent_id}: {e}")
            handle_exception(e, "Failed to remove agent.")

    async def remove_session(self, session_id: str):
        """Deletes a chat session and all its associated records."""
        async with self.oracle_session as session:
            repo = MemoryRepository(session)
            try:
                await repo.remove_context_by_id(session_id)
                logger.info(f"Successfully removed session {session_id}")
            except Exception as e:
                logger.error(f"Error removing session {session_id}: {e}")
                raise InternalServerError(f"Failed to delete session: {e}")
            
    async def get_conversation_list(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieves a list of all chat records associated with a specific `user_id`."""
        async with self.oracle_session as session:
            repo = MemoryRepository(session)
            return await repo.get_conversation_list_by_user_id(user_id)
        
    async def rename_conversation(self, session_id: str, new_title: str) -> None:
        """Renamesames a chat session title in the database."""
        async with self.oracle_session as session:
            repo = MemoryRepository(session)
            await repo.rename_conversation(session_id, new_title)
            logger.info(f"Successfully renamed session {session_id} to {new_title}")

        
    async def get_agent_model_params(self, agent_id: int) -> ModelParams:
        """
        获取智能体的模型参数

        Args:
            agent_id: 智能体ID

        Returns:
            模型参数
        """
        async with self.oracle_session as session:
            agent_repo = AgentRepository(session)
            try:
                agent = await agent_repo.get_by_id(agent_id)
                model_params = agent.models
                if not model_params:
                    raise NotFoundError(f"智能体 {agent_id} 没有配置模型参数")
                
                llm_params = {
                    "top_k": model_params.get("llm_top_k"),
                    "top_p": model_params.get("llm_top_p"),
                    "temperature": model_params.get("llm_temperature"),
                    "max_tokens": model_params.get("llm_max_tokens")
                }
                
                return ModelParams(
                    llm_model=model_params.get("llm_model", ""),
                    txt_embedding_model=model_params.get("txt_embedding_model", ""),
                    img_embedding_model=model_params.get("img_embedding_model", ""),
                    vlm_model=model_params.get("vlm_model", ""),
                    rerank_model=model_params.get("rerank_model", ""),
                    do_rerank=model_params.get("do_rerank", False),
                    llm_params=llm_params,
                    rerank_top_k=model_params.get("rerank_top_k"),
                )
            except Exception as e:
                handle_exception(e, "获取模型参数失败")

    async def get_kb_list(self, agent_id: int) -> list[int]:
        """
        获取智能体的知识库ID列表

        Args:
            agent_id: 智能体ID

        Returns:
            知识库配置列表
        """
        async with self.oracle_session as session:
            conf_repo = AgentConfRepository(session)
            try:
                confs = await conf_repo.get_by_agent(agent_id)
                kb_ids = []
                for c in confs:
                    if c.search_type == "hybrid":
                        kb_ids.append(c.kb_id)
                return kb_ids
            except Exception as e:
                handle_exception(e, "获取知识库配置列表失败")

    async def get_agent_profile(self, agent_id: int) -> int:
        """
        获取智能体的 profile ID

        Args:
            agent_id: 智能体ID

        Returns:
            profile ID
        """
        async with self.oracle_session as session:
            agent_repo = AgentRepository(session)
            try:
                agent = await agent_repo.get_by_id(agent_id)
                models = agent.models
                if not models:
                    raise NotFoundError(f"智能体 {agent_id} 没有配置 profile")
                profile = models.get("profile")
                if not profile:
                    raise NotFoundError(f"智能体 {agent_id} 没有配置 profile")
                return profile
            except Exception as e:
                handle_exception(e, "获取 profile 失败")