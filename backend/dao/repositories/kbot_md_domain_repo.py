from typing import Sequence, Optional

from sqlalchemy import select, delete
from backend.dao.entities.kbot_md_domain import KbotMdDomain
from backend.core.database.oracle import get_session

class KbotMdDomainRepository:
    """Repository for KBOT_MD_DOMAIN table operations."""
    
    async def create(self, domain: KbotMdDomain) -> KbotMdDomain:
        """Create a new domain record."""
        async with get_session() as session:
            session.add(domain)
            await session.commit()
            await session.refresh(domain)
            return domain
    
    async def get_by_id(self, domain_id: int) -> Optional[KbotMdDomain]:
        """Get domain by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDomain).where(KbotMdDomain.domain_id == domain_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdDomain]:
        """Get all domain records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdDomain))
            return result.scalars().all()
    
    async def update(self, domain: KbotMdDomain) -> KbotMdDomain:
        """Update a domain record."""
        async with get_session() as session:
            session.add(domain)
            await session.commit()
            await session.refresh(domain)
            return domain
    
    async def delete(self, domain_id: int) -> bool:
        """Delete a domain record by ID."""
        async with get_session() as session:
            domain = await self.get_by_id(domain_id)
            if not domain:
                return False
            await session.delete(domain)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdDomain]:
        """Get domains by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDomain).where(KbotMdDomain.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_name(self, name: str) -> Sequence[KbotMdDomain]:
        """Get domains by name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDomain).where(KbotMdDomain.name == name)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: str) -> Sequence[KbotMdDomain]:
        """Get domains by status."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDomain).where(KbotMdDomain.status == status)
            )
            return result.scalars().all()
    
    async def get_by_app_and_name(self, app_id: int, name: str) -> Optional[KbotMdDomain]:
        """Get domain by app_id and name (unique constraint)."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDomain)
                .where(KbotMdDomain.app_id == app_id)
                .where(KbotMdDomain.name == name)
            )
            return result.scalars().first()