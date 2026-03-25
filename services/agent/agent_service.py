from loguru import logger

from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.exceptions import *
from dao.repositories import (AgentRepository, AgentConfRepository,
                             PromptRepository, FileRepository)
from services.search.rerank import TxtBaseRerank
from services.search.kb_search import TxtBaseSearch
from utils.clients import AIModelClient
from services.ai_model import AIModelService



class AgentService:
    """Agent service class for managing knowledge base search and AI chat interactions."""

    @property
    def oracle_session(self):
        """Provides a database session instance."""
        return get_session()

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
                await agent_repo.delete(agent_id)
                
                sess_repo = ChatSessionRepository(session)
                sessions = await sess_repo.get_by_agent(agent_id)
                await ChatMemoryRepository(session).delete_by_ids([s.id for s in sessions])
                await sess_repo.delete_by_agent_id(agent_id)
                logger.info(f"Successfully removed agent {agent_id} and related records.")
        except Exception as e:
            logger.error(f"Error removing agent {agent_id}: {e}")
            handle_exception(e, "Failed to remove agent.")

    async def feedback(self, memory_id: int, feedback: int):
        """Updates user feedback for a specific chat record."""
        async with self.oracle_session as session:
            logger.info(f"Submitting feedback {feedback} for memory {memory_id}")
            await ChatMemoryRepository(session).feedback(memory_id, feedback)

    async def get_session_history(self, session_id: str):
        """Retrieves history for a specific chat session."""
        async with self.oracle_session as session:
            return await ChatMemoryRepository(session).get_session_history(session_id)

    async def remove_session(self, session_id: str):
        """Deletes a chat session and all its associated records."""
        async with self.oracle_session as session:
            try:
                await ChatMemoryRepository(session).delete_session_records(session_id)
                await ChatSessionRepository(session).delete(session_id)
                logger.info(f"Successfully removed session {session_id}")
            except Exception as e:
                logger.error(f"Error removing session {session_id}: {e}")
                raise InternalServerError(f"Failed to delete session: {e}")
      