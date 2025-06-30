from typing import Sequence, Optional
from sqlalchemy import select, delete, and_
from backend.dao.entities.kbot_md_sys_conf import (
    KbotMdSysConf, 
    ParamType
)
from backend.core.database.meta_oracle import get_session

class KbotMdSysConfRepository:
    """Repository for KBOT_MD_SYS_CONF table operations."""
    
    async def create(self, config: KbotMdSysConf) -> KbotMdSysConf:
        """Create a new system configuration record."""
        async with get_session() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
            return config
    
    async def get_by_id(self, conf_id: int) -> Optional[KbotMdSysConf]:
        """Get system configuration by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf).where(KbotMdSysConf.conf_id == conf_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdSysConf]:
        """Get all system configuration records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdSysConf))
            return result.scalars().all()
    
    async def update(self, config: KbotMdSysConf) -> KbotMdSysConf:
        """Update a system configuration record."""
        async with get_session() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
            return config
    
    async def delete(self, conf_id: int) -> bool:
        """Delete a system configuration record by ID."""
        async with get_session() as session:
            config = await self.get_by_id(conf_id)
            if not config:
                return False
            await session.delete(config)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdSysConf]:
        """Get system configurations by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf).where(KbotMdSysConf.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_param_type(self, param_type: ParamType) -> Sequence[KbotMdSysConf]:
        """Get system configurations by parameter type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf).where(KbotMdSysConf.param_type == param_type.value)
            )
            return result.scalars().all()
    
    async def get_by_name(self, param_name: str) -> Optional[KbotMdSysConf]:
        """Get system configuration by parameter name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf).where(KbotMdSysConf.param_name == param_name)
            )
            return result.scalars().first()
    
    async def get_by_name_and_type(self, param_name: str, param_type: ParamType) -> Optional[KbotMdSysConf]:
        """Get system configuration by name and type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf)
                .where(and_(
                    KbotMdSysConf.param_name == param_name,
                    KbotMdSysConf.param_type == param_type.value
                ))
            )
            return result.scalars().first()