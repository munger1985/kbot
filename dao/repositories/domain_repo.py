from typing import Sequence
from sqlalchemy import select
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import DomainEntity
from core.dictionary import Status
from .base_repo import BaseRepository


class DomainRepository(BaseRepository[DomainEntity]):
    """Repository for KBOT_MD_DOMAIN table operations."""
    
    async def get_by_id(self, domain_id: int) -> DomainEntity:
        """Get domain by ID."""
        try:
            stmt = select(DomainEntity).where(DomainEntity.domain_id == domain_id)
            result = await self.session.execute(stmt)
            domain = result.scalar_one_or_none()
            
            if not domain:
                raise DataNotFoundException(f"Domain {domain_id} does not exist")
            
            return domain
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get domain by ID", original_error=e)
    
    async def get_by_name(self, name: str) -> Sequence[DomainEntity]:
        """Get domains by name."""
        try:
            stmt = select(DomainEntity).where(DomainEntity.name == name)
            result = await self.session.execute(stmt)
            domains = result.scalars().all()
            
            return domains
        except Exception as e:
            raise DatabaseException("Failed to get domains by name", original_error=e)
    
    async def get_by_status(self, status: Status) -> Sequence[DomainEntity]:
        """Get domains by status."""
        try:
            stmt = select(DomainEntity).where(DomainEntity.status == status.value)
            result = await self.session.execute(stmt)
            domains = result.scalars().all()
            
            return domains
        except Exception as e:
            raise DatabaseException("Failed to get domains by status", original_error=e)