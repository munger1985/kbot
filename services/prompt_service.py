from loguru import logger
from core.database.oracle import get_session
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

    async def get_prompt_by_unique_name(self, unique_name: str) -> str | None:
        """Gets the prompt content by unique name.

        Args:
            unique_name: The unique name of the prompt.

        Returns:
            The prompt content as a string.
        """
        async with self.oracle_session as session:
            repo = PromptRepository(session)
            try:
                return await repo.get_prompt_by_unique_name(unique_name)
            except Exception as e:
                logger.error(f"Failed to get prompt by unique name '{unique_name}', original_error={e}")
                return None
            
    async def get_prompt_by_id(self, prompt_id: int) -> str | None:
        """Gets the prompt content by id.

        Args:
            prompt_id: The id of the prompt.

        Returns:
            The prompt content as a string.
        """
        async with self.oracle_session as session:
            repo = PromptRepository(session)
            try:
                return await repo.get_prompt_by_id(prompt_id)
            except Exception as e:
                logger.error(f"Failed to get prompt by id '{prompt_id}', original_error={e}")
                return None
