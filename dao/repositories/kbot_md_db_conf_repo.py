from typing import Sequence
from sqlalchemy import select
from sqlalchemy import func
from dao.entities.kbot_md_db_conf import KbotMdDbConf
from dao.entities.kbot_md_kb import KbotMdKb
from dao.data_dict import ( 
    DbType,
    Status
)
from core.database.meta_oracle import get_session


class KbotMdDbConfRepository:
    """Repository for KBOT_MD_DB_CONF table operations."""
    
    async def create(self, db_conf: KbotMdDbConf) -> KbotMdDbConf:
        """Create a new database configuration record."""
        async with get_session() as session:
            session.add(db_conf)
            await session.commit()
            await session.refresh(db_conf)
            return db_conf
    
    async def get_by_id(self, db_id: int) -> KbotMdDbConf | None:
        """Get database configuration by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.db_id == db_id)
            )
            return result.scalar_one_or_none()
    
    async def get_by_kbid(self, kb_id: int) -> KbotMdDbConf | None:
        """Get database configuration by KB ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf)
                .join(KbotMdKb, KbotMdKb.db_conn_id == KbotMdDbConf.db_id)
                .where(KbotMdKb.kb_id == kb_id)
            )
            return result.scalars().first()
        
    async def get_all(self) -> Sequence[KbotMdDbConf]:
        """Get all database configuration records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdDbConf))
            return result.scalars().all()
    
    async def update(self, db_conf: KbotMdDbConf) -> KbotMdDbConf:
        """Update a database configuration record."""
        async with get_session() as session:
            session.add(db_conf)
            await session.commit()
            await session.refresh(db_conf)
            return db_conf
    
    async def delete(self, db_id: int) -> bool:
        """Delete a database configuration record by ID."""
        async with get_session() as session:
            db_conf = await self.get_by_id(db_id)
            if not db_conf:
                return False
            await session.delete(db_conf)
            await session.commit()
            return True
    
    async def get_by_db_type(self, db_type: DbType) -> Sequence[KbotMdDbConf]:
        """Get database configurations by database type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.db_type == db_type.value)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: Status) -> Sequence[KbotMdDbConf]:
        """Get database configurations by status."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.status == status.value)
            )
            return result.scalars().all()
    
    
    