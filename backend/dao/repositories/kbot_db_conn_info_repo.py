from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.dao.entities.kbot_db_conn_info import KBotDbConnInfo
from backend.core.database.oracle import async_session

class KBotDbConnInfoRepository:
    """Repository for KBOT_DB_CONN_INFO table operations."""
    
    async def create(self, conn_info: KBotDbConnInfo) -> KBotDbConnInfo:
        """Create a new connection info record."""
        async with async_session() as session:
            session.add(conn_info)
            await session.commit()
            await session.refresh(conn_info)
            return conn_info
    
    async def get_by_id(self, id: int) -> Optional[KBotDbConnInfo]:
        """Get connection info by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotDbConnInfo).where(KBotDbConnInfo.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotDbConnInfo]:
        """Get all connection info records."""
        async with async_session() as session:
            result = await session.execute(select(KBotDbConnInfo))
            return result.scalars().all()
    
    async def update(self, conn_info: KBotDbConnInfo) -> KBotDbConnInfo:
        """Update a connection info record."""
        async with async_session() as session:
            session.add(conn_info)
            await session.commit()
            await session.refresh(conn_info)
            return conn_info
    
    async def delete(self, id: int) -> bool:
        """Delete a connection info record by ID."""
        async with async_session() as session:
            conn_info = await self.get_by_id(id)
            if conn_info:
                await session.delete(conn_info)
                await session.commit()
                return True
            return False