from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_sys_conf import KBotSysConf
from backend.core.database.oracle import async_session

class KBotSysConfRepository:
    """Repository for KBOT_SYS_CONF table operations."""
    
    async def create(self, config: KBotSysConf) -> KBotSysConf:
        """Create a new system configuration record."""
        async with async_session() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
            return config
    
    async def get_by_id(self, id: int) -> Optional[KBotSysConf]:
        """Get configuration by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotSysConf).where(KBotSysConf.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotSysConf]:
        """Get all configuration records."""
        async with async_session() as session:
            result = await session.execute(select(KBotSysConf))
            return result.scalars().all()
    
    async def update(self, config: KBotSysConf) -> KBotSysConf:
        """Update a configuration record."""
        async with async_session() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
            return config
    
    async def delete(self, id: int) -> bool:
        """Delete a configuration record by ID."""
        async with async_session() as session:
            config = await self.get_by_id(id)
            if config:
                await session.delete(config)
                await session.commit()
                return True
            return False
    
    async def get_by_app_id(self, app_id: int) -> List[KBotSysConf]:
        """Get configurations by APP ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotSysConf).where(KBotSysConf.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_param_name(self, param_name: str) -> List[KBotSysConf]:
        """Get configurations by parameter name."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotSysConf).where(KBotSysConf.param_name == param_name)
            )
            return result.scalars().all()
    
    async def get_by_name_and_value(self, param_name: str, param_value: str) -> List[KBotSysConf]:
        """Get configurations by parameter name and value."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotSysConf)
                .where(KBotSysConf.param_name == param_name)
                .where(KBotSysConf.param_value == param_value)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: str) -> List[KBotSysConf]:
        """Get configurations by status."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotSysConf).where(KBotSysConf.status == status)
            )
            return result.scalars().all()