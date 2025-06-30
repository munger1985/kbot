from typing import Sequence, Optional
from sqlalchemy import select, delete
from backend.dao.entities.kbot_md_kb import KbotMdKb, KbCategory, KbStatus
from backend.dao.entities.kbot_md_db_conf import KbotMdDbConf
from backend.core.database.meta_oracle import get_session

class KbotMdKbRepository:
    """Repository for KBOT_MD_KB table operations."""
    
    async def create(self, kb: KbotMdKb) -> KbotMdKb:
        """Create a new knowledge base record."""
        async with get_session() as session:
            session.add(kb)
            await session.commit()
            await session.refresh(kb)
            return kb
    
    async def get_by_id(self, kb_id: int) -> Optional[KbotMdKb]:
        """Get knowledge base by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.kb_id == kb_id)
            )
            return result.scalars().first()
    
    async def get_by_name(self, kb_name: str) -> Optional[KbotMdKb]:
        """Get knowledge base by name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.kb_name == kb_name)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdKb]:
        """Get all knowledge base records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKb))
            return result.scalars().all()
    
    async def update(self, kb: KbotMdKb) -> KbotMdKb:
        """Update a knowledge base record."""
        async with get_session() as session:
            session.add(kb)
            await session.commit()
            await session.refresh(kb)
            return kb
    
    async def delete(self, kb_id: int) -> bool:
        """Delete a knowledge base record by ID."""
        async with get_session() as session:
            kb = await self.get_by_id(kb_id)
            if not kb:
                return False
            await session.delete(kb)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdKb]:
        """Get knowledge bases by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_domain_id(self, domain_id: int) -> Sequence[KbotMdKb]:
        """Get knowledge bases by domain ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.domain_id == domain_id)
            )
            return result.scalars().all()
    
    async def get_by_category(self, category: KbCategory) -> Sequence[KbotMdKb]:
        """Get knowledge bases by category."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.kb_category == category.value)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: KbStatus) -> Sequence[KbotMdKb]:
        """Get knowledge bases by status."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb).where(KbotMdKb.kb_status == status.value)
            )
            return result.scalars().all()
    
    async def get_by_app_domain_name(self, app_id: int, domain_id: int, kb_name: str) -> Optional[KbotMdKb]:
        """Get knowledge base by app_id, domain_id and kb_name (unique constraint)."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKb)
                .where(KbotMdKb.app_id == app_id)
                .where(KbotMdKb.domain_id == domain_id)
                .where(KbotMdKb.kb_name == kb_name)
            )
            return result.scalars().first()
    
    async def get_dbconf_by_kbid(self, kbid: int) -> Optional[KbotMdDbConf]:
        """Get database configuration by knowledge base ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf)
                .join(KbotMdKb, KbotMdKb.db_conn_id == KbotMdDbConf.db_id)
                .where(KbotMdKb.kb_id == kbid)
                .where(KbotMdDbConf.status == 'Y')
            )
            return result.scalars().first()