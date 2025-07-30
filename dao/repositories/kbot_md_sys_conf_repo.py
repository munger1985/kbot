from typing import Sequence
from sqlalchemy import select
from dao.entities.kbot_md_sys_conf import KbotMdSysConf
from core.database.meta_oracle import get_session


class KbotMdSysConfRepository:
    """Repository for KBOT_MD_SYS_CONF table operations."""
    
    async def create(self, config: KbotMdSysConf) -> KbotMdSysConf:
        """Create a new system configuration record."""
        async with get_session() as session:
            session.add(config)
            await session.commit()
            await session.refresh(config)
            return config
    
    async def get_by_id(self, conf_id: int) -> KbotMdSysConf | None:
        """Get system configuration by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdSysConf).where(KbotMdSysConf.conf_id == conf_id)
            )
            return result.scalar_one_or_none()
    
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
