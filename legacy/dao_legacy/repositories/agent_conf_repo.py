from sqlalchemy import select, update, delete, and_
from platform_core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import AgentConfEntity
from .base_repo import BaseRepository


class AgentConfRepository(BaseRepository[AgentConfEntity]):
    """Repository class for KBOT_MD_AGENT_CONF table"""

    async def create(self, agent_conf: AgentConfEntity) -> None:
        """创建智能体配置"""
        try:
            self.session.add(agent_conf)
        except Exception as e:
            raise DatabaseException("创建智能体配置失败", original_error=e)

    async def update(self, agent_conf_id: int, **kwargs) -> None:
        """根据ID更新智能体配置"""
        try:
            result = await self.session.execute(
                update(AgentConfEntity)
                .where(AgentConfEntity.id == agent_conf_id)
                .values(**kwargs)
                .returning(AgentConfEntity.id)
            )
            if result.scalar() is None:
                raise DataNotFoundException(f"智能体配置 {agent_conf_id} 不存在")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"更新智能体配置失败", original_error=e)

    async def delete(self, agent_conf_id: int) -> None:
        """根据ID删除智能体配置"""
        try:
            await self.session.execute(
                delete(AgentConfEntity)
                .where(AgentConfEntity.id == agent_conf_id)
            )
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"删除智能体配置失败", original_error=e)

    async def get(self, agent_conf_id: int) -> AgentConfEntity:
        """获取智能体配置"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity).where(AgentConfEntity.id == agent_conf_id)
            )
            agent_conf = result.scalar_one_or_none()
            if not agent_conf:
                raise DataNotFoundException(f"智能体配置 {agent_conf_id} 不存在")
            return agent_conf
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取智能体配置失败", original_error=e)

    async def get_all(self) -> list[AgentConfEntity]:
        """获取所有智能体配置"""
        try:
            result = await self.session.execute(select(AgentConfEntity))
            agent_confs = result.scalars().all()
            if not agent_confs or len(agent_confs) == 0:
                raise DataNotFoundException("未找到智能体配置")
            return list(agent_confs)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("获取所有智能体配置失败", original_error=e)

    async def get_by_agent_and_kb(self, agent_id: int, kb_id: int) -> AgentConfEntity:
        """根据助手ID和知识库ID获取配置"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity)
                .where(
                    and_(
                        AgentConfEntity.agent_id == agent_id,
                        AgentConfEntity.tool_id == kb_id
                    )
                )
            )
            agent_conf = result.scalar_one_or_none()
            if not agent_conf:
                raise DataNotFoundException(f"智能体配置 {agent_id} {kb_id} 不存在")
            return agent_conf
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("获取智能体配置失败", original_error=e)

    async def get_by_agent(self, agent_id: int) -> list[AgentConfEntity]:
        """根据助手ID获取所有配置"""
        try:
            result = await self.session.execute(
                select(AgentConfEntity)
                .where(AgentConfEntity.agent_id == agent_id)
            )
            agent_confs = result.scalars().all()
            if not agent_confs or len(agent_confs) == 0:
                raise DataNotFoundException(f"智能体 {agent_id} 不存在配置")
            return list(agent_confs)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("获取智能体配置失败", original_error=e)

    async def delete_by_agent_id(self, agent_id: int) -> None:
        """根据助手ID删除配置"""
        try:
            await self.session.execute(
                delete(AgentConfEntity)
                .where(AgentConfEntity.agent_id == agent_id)
            )
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("删除智能体配置失败", original_error=e)
        
    
