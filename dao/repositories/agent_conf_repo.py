from sqlalchemy import select, update, delete, and_
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import AgentConfEntity
from .base_repo import BaseRepository


class AgentConfRepository(BaseRepository[AgentConfEntity]):
    """Repository class for KBOT_MD_AGENT_CONF table"""

    async def get_by_id(self, conf_id: int) -> AgentConfEntity:
        """Get agent configuration by ID"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity).where(AgentConfEntity.conf_id == conf_id)
            )
            agent_conf = result.scalar_one_or_none()
            if not agent_conf:
                raise DataNotFoundException(f"Agent configuration {conf_id} does not exist")
            return agent_conf
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get agent configuration", original_error=e)

    async def get_by_agent_id(self, agent_id: int) -> list[AgentConfEntity]:
        """Get all agent configurations by agent ID"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity).where(AgentConfEntity.agent_id == agent_id)
            )
            agent_confs = result.scalars().all()
            if not agent_confs or len(agent_confs) == 0:
                raise DataNotFoundException(f"No agent configurations found for agent {agent_id}")
            return list(agent_confs)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get agent configurations", original_error=e)

    async def get_by_agent_and_kb(self, agent_id: int, kb_id: int) -> AgentConfEntity:
        """Get agent configuration by agent ID and kb ID"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity).where(
                    and_(
                        AgentConfEntity.agent_id == agent_id,
                        AgentConfEntity.tool_id == kb_id,
                        AgentConfEntity.search_type == 1
                    )
                )
            )
            agent_conf = result.scalar_one_or_none()
            if not agent_conf:
                raise DataNotFoundException(f"Agent configuration not found for agent {agent_id} and kb {kb_id}")
            return agent_conf
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get agent configuration", original_error=e)

    async def delete(self, conf_id: int) -> None:
        """Delete agent configuration by ID"""
        try:
            result = await self.session.execute(
                delete(AgentConfEntity).where(AgentConfEntity.conf_id == conf_id)
            )
        except Exception as e:
            raise DatabaseException("Failed to delete agent configuration", original_error=e)

    async def delete_by_agent_id(self, agent_id: int) -> None:
        """Delete agent configurations by agent ID"""
        try:
            result = await self.session.execute(
                delete(AgentConfEntity).where(AgentConfEntity.agent_id == agent_id)
            )
        except Exception as e:
            raise DatabaseException("Failed to delete agent configurations", original_error=e)

    async def get_all(self) -> list[AgentConfEntity]:
        """Get all agent configurations"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity).order_by(AgentConfEntity.conf_id)
            )
            agent_confs = result.scalars().all()
            if not agent_confs or len(agent_confs) == 0:
                raise DataNotFoundException("No agent configurations found")
            return list(agent_confs)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get all agent configurations", original_error=e)