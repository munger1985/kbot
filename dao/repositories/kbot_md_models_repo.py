from typing import Sequence, Optional
from sqlalchemy import select, delete, and_
from dao.entities.kbot_md_models import KbotMdModels
from dao.data_dict import ModelCategory, Status
from core.database.meta_oracle import get_session


class KbotMdModelsRepository:
    """Repository for KBOT_MD_KB_MODELS table operations."""
    
    async def get_all_embedding_models(self) -> Sequence[KbotMdModels]:
        """
        获取所有嵌入模型
        
        Returns:
            Sequence[KbotMdModels]: 嵌入模型列表
        """
        async with get_session() as session:
            query = select(KbotMdModels).where(
                and_(
                    KbotMdModels.category == ModelCategory.EMBEDDING.value,
                    KbotMdModels.status == Status.ENABLED.value  # 只获取启用状态的模型
                )
            )
            result = await session.execute(query)
            return result.scalars().all()
    
    async def create(self, model: KbotMdModels) -> KbotMdModels:
        """Create a new knowledge base model record."""
        async with get_session() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
    
    async def get_by_id(self, model_id: int) -> Optional[KbotMdModels]:
        """Get knowledge base model by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.model_id == model_id)
            )
            return result.scalars().first()
    
    async def get_all(self) -> Sequence[KbotMdModels]:
        """Get all knowledge base model records."""
        async with get_session() as session:
            result = await session.execute(select(KbotMdModels))
            return result.scalars().all()
    
    async def update(self, model: KbotMdModels) -> KbotMdModels:
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
    
    async def get_by_app_id(self, app_id: int) -> Sequence[KbotMdModels]:
        """Get knowledge base models by application ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.app_id == app_id)
            )
            return result.scalars().all()
      
    async def get_by_provider(self, provider: str) -> Sequence[KbotMdModels]:
        """Get knowledge base models by provider."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.provider == provider)
            )
            return result.scalars().all()    
    