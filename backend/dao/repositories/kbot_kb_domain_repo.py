from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_kb_domain import KBotKBDomain
from backend.core.database.oracle import async_session

class KBotKBDomainRepository:
    """Repository for KBOT_KB_DOMAIN table operations."""
    
    async def create(self, domain: KBotKBDomain) -> KBotKBDomain:
        """Create a new KB domain record."""
        async with async_session() as session:
            session.add(domain)
            await session.commit()
            await session.refresh(domain)
            return domain
    
    async def get_by_id(self, id: int) -> Optional[KBotKBDomain]:
        """Get KB domain by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBDomain).where(KBotKBDomain.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotKBDomain]:
        """Get all KB domain records."""
        async with async_session() as session:
            result = await session.execute(select(KBotKBDomain))
            return result.scalars().all()
    
    async def update(self, domain: KBotKBDomain) -> KBotKBDomain:
        """Update a KB domain record."""
        async with async_session() as session:
            session.add(domain)
            await session.commit()
            await session.refresh(domain)
            return domain
    
    async def delete(self, id: int) -> bool:
        """Delete a KB domain record by ID."""
        async with async_session() as session:
            domain = await self.get_by_id(id)
            if domain:
                await session.delete(domain)
                await session.commit()
                return True
            return False
    
    async def get_by_app_id(self, app_id: int) -> List[KBotKBDomain]:
        """Get domains by APP ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotKBDomain).where(KBotKBDomain.app_id == app_id)
            )
            return result.scalars().all()