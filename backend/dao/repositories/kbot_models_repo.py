from typing import List, Optional

from sqlalchemy import select

from backend.dao.entities.kbot_models import KBotModels
from backend.core.database.oracle import async_session

class KBotKBModelsRepository:
    """Repository for KBOT_KB_MODELS table operations."""
    
    async def create(self, model: KBotModels) -> KBotModels:
        """Create a new KB model record."""
        async with async_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
    
    async def get_by_id(self, id: int) -> Optional[KBotModels]:
        """Get KB model by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotModels).where(KBotModels.id == id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> List[KBotModels]:
        """Get all KB model records."""
        async with async_session() as session:
            result = await session.execute(select(KBotModels))
            return result.scalars().all()
    
    async def update(self, model: KBotModels) -> KBotModels:
        """Update a KB model record."""
        async with async_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
    
    async def delete(self, id: int) -> bool:
        """Delete a KB model record by ID."""
        async with async_session() as session:
            model = await self.get_by_id(id)
            if model:
                await session.delete(model)
                await session.commit()
                return True
            return False
    
    async def get_by_app_id(self, app_id: int) -> List[KBotModels]:
        """Get models by APP ID."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotModels).where(KBotModels.app_id == app_id)
            )
            return result.scalars().all()
    
    async def get_by_status(self, status: str) -> List[KBotModels]:
        """Get models by status."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotModels).where(KBotModels.status == status)
            )
            return result.scalars().all()
    
    async def get_by_provider(self, provider: str) -> List[KBotModels]:
        """Get models by provider."""
        async with async_session() as session:
            result = await session.execute(
                select(KBotModels).where(KBotModels.provider == provider)
            )
            return result.scalars().all()