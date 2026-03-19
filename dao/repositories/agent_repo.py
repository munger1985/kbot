from typing import Sequence
from sqlalchemy import select, update, delete
from loguru import logger
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import AgentEntity
from .base_repo import BaseRepository

class AgentRepository(BaseRepository[AgentEntity]):
    """Agent metadata repository"""
    
    async def create(self, agent: AgentEntity) -> AgentEntity:
        """Create agent metadata"""
        try:
            self.session.add(agent)
            await self.session.flush()
            await self.session.refresh(agent)
            return agent
        except Exception as e:
            raise DatabaseException("Failed to create agent metadata", original_error=e)

    async def get_by_id(self, agent_id: int) -> AgentEntity:
        """Get agent metadata by id"""
        try:
            logger.debug(f"[AgentRepo] Attempting to get agent by id: {agent_id}")
            stmt = select(AgentEntity).where(AgentEntity.agent_id == agent_id)
            result = await self.session.execute(stmt)
            agent = result.scalar_one_or_none()
            if not agent:
                logger.warning(f"[AgentRepo] Agent metadata {agent_id} does not exist in database")
                raise DataNotFoundException(f"Agent metadata {agent_id} does not exist")
            logger.debug(f"[AgentRepo] Successfully retrieved agent {agent_id}: {agent.agent_name}")
            return agent
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"[AgentRepo] Failed to get agent metadata by id={agent_id}, error type: {type(e).__name__}, error: {str(e)}", exc_info=True)
            raise DatabaseException("Failed to get agent metadata by id", original_error=e)

    async def get_all(self) -> Sequence[AgentEntity]:
        """Get all agent metadata"""
        try:
            stmt = select(AgentEntity)
            result = await self.session.execute(stmt)
            agents = result.scalars().all()
            if not agents or len(agents) == 0:
                raise DataNotFoundException("No agent metadata found")
            return agents
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get all agent metadata", original_error=e)

    async def update(self, agent_id: int, **kwargs) -> AgentEntity:
        """Update agent metadata by id"""
        try:
            stmt = (
                update(AgentEntity)
                .where(AgentEntity.agent_id == agent_id)
                .values(** kwargs)
                .returning(AgentEntity)
            )
            result = await self.session.execute(stmt)
            updated_agent = result.scalar_one_or_none()
            
            if not updated_agent:
                raise DataNotFoundException(f"Agent metadata {agent_id} does not exist")

            return updated_agent
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to update agent metadata", original_error=e)

    async def delete(self, agent_id: int) -> None:
        """Delete agent metadata by id"""
        try:
            stmt = delete(AgentEntity).where(AgentEntity.agent_id == agent_id)
            result = await self.session.execute(stmt)
        except Exception as e:
            raise DatabaseException("Failed to delete agent metadata", original_error=e)
    
    async def get_app_id(self, agent_id: int) -> int:
        """Get app id by agent id"""
        try:
            stmt = (
                select(AgentEntity.app_id)
                .where(AgentEntity.agent_id == agent_id)
            )
            result = await self.session.execute(stmt)
            app_id = result.scalar_one_or_none()
            
            if app_id is None:
                raise DataNotFoundException(f"App id not found for agent {agent_id}")
            
            return app_id
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get app id by agent id", original_error=e)
    
    async def get_prompt(self, agent_id: int) -> int:
        """Get prompt id by agent id"""
        try:
            stmt = (
                select(AgentEntity.prompt_id)
                .where(AgentEntity.agent_id == agent_id)
            )
            result = await self.session.execute(stmt)
            prompt_id = result.scalar_one_or_none()
            
            if prompt_id is None:
                raise DataNotFoundException(f"Prompt id not found for agent {agent_id}")
            
            return prompt_id
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get prompt id by agent id", original_error=e)