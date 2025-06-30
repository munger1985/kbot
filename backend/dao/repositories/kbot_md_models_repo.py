from typing import Sequence, Optional
from sqlalchemy import select, delete, and_
from backend.dao.entities.kbot_md_models import KbotMdKbModels
from backend.core.database.oracle import get_session

class KbotMdKbModelsRepository:
    """Repository for KBOT_MD_KB_MODELS table operations."""
    
    async def create(self, model: KbotMdKbModels) -> KbotMdKbModels:
        """Create a new knowledge base model record."""
        async with get_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
    
    async def get_by_id(self, model_id: int) -> Optional[KbotMdKbModels]:
        """Get knowledge base model by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels).where(KbotMdKbModels.model_id == model_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdKbModels]:
        """Get all knowledge base model records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdKbModels))
            return result.scalars().all()
    
    async def update(self, model: KbotMdKbModels) -> KbotMdKbModels:
        """Update a knowledge base model record."""
        async with get_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
    
    async def delete(self, model_id: int) -> bool:
        """Delete a knowledge base model record by ID."""
        async with get_session() as session:
            model = await self.get_by_id(model_id)
            if not model:
                return False
            await session.delete(model)
            await session.commit()
            return True
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdKbModels]:
        """Get knowledge base models by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels).where(KbotMdKbModels.app_id == app_id)
            )
            return result.scalars().all()
      
    async def get_by_provider(self, provider: str) -> Sequence[KbotMdKbModels]:
        """Get knowledge base models by provider."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels).where(KbotMdKbModels.provider == provider)
            )
            return result.scalars().all()    
    
    async def get_by_name(self, model_name: str) -> Optional[KbotMdKbModels]:
        """Get knowledge base model by technical name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels).where(KbotMdKbModels.model_name == model_name)
            )
            return result.scalars().first()
    
    async def get_by_display_name(self, display_name: str) -> Optional[KbotMdKbModels]:
        """Get knowledge base model by display name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels).where(KbotMdKbModels.display_name == display_name)
            )
            return result.scalars().first()
    
    async def get_by_name_and_provider(self, model_name: str, provider: str) -> Optional[KbotMdKbModels]:
        """Get knowledge base model by name and provider."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdKbModels)
                .where(and_(
                    KbotMdKbModels.model_name == model_name,
                    KbotMdKbModels.provider == provider
                ))
            )
            return result.scalars().first()