from typing import Sequence, Optional
from sqlalchemy import select, delete, and_
from sqlalchemy import func
from backend.dao.entities.kbot_md_db_conf import (
    KbotMdDbConf, 
    DbType,
    DbStatus
)
from backend.core.database.oracle import get_session

class KbotMdDbConfRepository:
    """Repository for KBOT_MD_DB_CONF table operations."""
    
    async def create(self, db_conf: KbotMdDbConf) -> KbotMdDbConf:
        """Create a new database configuration record."""
        async with get_session() as session:
            session.add(db_conf)
            await session.commit()
            await session.refresh(db_conf)
            return db_conf
    
    async def get_by_id(self, db_id: int) -> Optional[KbotMdDbConf]:
        """Get database configuration by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.db_id == db_id)
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
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdDbConf]:
        """Get database configurations by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_db_type(self, db_type: DbType) -> Sequence[KbotMdDbConf]:
        """Get database configurations by database type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.db_type == db_type.value)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: DbStatus) -> Sequence[KbotMdDbConf]:
        """Get database configurations by status."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.status == status.value)
            )
            return result.scalars().all()
    
    async def get_by_display_name(self, display_name: str) -> Optional[KbotMdDbConf]:
        """Get database configuration by display name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(KbotMdDbConf.db_display_name == display_name)
            )
            return result.scalars().first()
    
    async def get_by_conn_str_property(self, property_name: str, property_value: str) -> Sequence[KbotMdDbConf]:
        """Get database configurations by connection string property."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(
                    func.jsonb_extract_path_text(KbotMdDbConf.db_conn_str, property_name) == property_value
                )
            )
            return result.scalars().all()
    
    async def get_by_app_and_type(self, app_id: int, db_type: DbType) -> Sequence[KbotMdDbConf]:
        """Get database configurations by application ID and database type."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdDbConf).where(
                    and_(
                        KbotMdDbConf.app_id == app_id,
                        KbotMdDbConf.db_type == db_type.value
                    )
                )
            )
            return result.scalars().all()