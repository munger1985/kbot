from loguru import logger
from typing import Any
from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.exceptions import *
from dao.repositories import (AgentRepository, AgentConfRepository,
                             PromptRepository, FileRepository, MemoryEntryRepository)



class AgentService:
    """Agent service class for managing knowledge base search and AI chat interactions."""

    @property
    def oracle_session(self):
        """Provides a database session instance."""
        return get_session()
    
    async def feedback(self, entry_id: int, feedback: int):
        """Updates user feedback for a specific chat record."""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            await repo.update_feedback(entry_id, feedback)
            logger.info(f"Updated feedback {feedback} for memory {entry_id}")

    async def get_context_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieves history for a specific chat session."""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
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
                await agent_repo.delete(agent_id)
                
                sess_repo = MemoryEntryRepository(session)
                await sess_repo.remove_context_by_agent(agent_id)
                logger.info(f"Successfully removed agent {agent_id} and related sessions.")
        except Exception as e:
            logger.error(f"Error removing agent {agent_id}: {e}")
            handle_exception(e, "Failed to remove agent.")

    async def remove_session(self, session_id: str):
        """Deletes a chat session and all its associated records."""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            try:
                await repo.remove_context_by_id(session_id)
                logger.info(f"Successfully removed session {session_id}")
            except Exception as e:
                logger.error(f"Error removing session {session_id}: {e}")
                raise InternalServerError(f"Failed to delete session: {e}")
      