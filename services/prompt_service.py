from loguru import logger
from core.database.oracle import get_session
from dao.entities import AgentConfEntity
from dao.repositories import PromptRepository, AgentRepository
from core.exceptions import *


class PromptService:
    """Prompt service class."""
    
    def __init__(self):
        """Initializes the prompt service."""

    @property
    def oracle_session(self):
        return get_session()
    

    async def remove_prompt_by_agent(self, agent_id: int):
        """Removes all prompts for the agent.

        Args:
            agent_id: Agent ID.
        """
        try:
            async with self.oracle_session as session:
                agent_repo = AgentRepository(session)
                agent = await agent_repo.get_by_id(agent_id)
                if agent.prompt_id:
                    prompt_repo = PromptRepository(session)
                    await prompt_repo.delete(agent.prompt_id)
                    logger.info(f"Prompt {agent.prompt_id} removed for agent {agent_id}.")
        except Exception as e:
            handle_exception(e, "Failed to remove prompts for agent.")