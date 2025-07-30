from typing import Sequence
from sqlalchemy import select, and_
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
        
    async def get_all_llm_models(self) -> Sequence[KbotMdModels]:
        """
        获取所有LLM模型
        
        Returns:
            Sequence[KbotMdModels]: LLM模型列表
        """
        async with get_session() as session:
            query = select(KbotMdModels).where(
                and_(
                    KbotMdModels.category == ModelCategory.LLM.value,
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
    
    async def get_by_id(self, model_id: int) -> KbotMdModels | None:
        """Get knowledge base model by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.model_id == model_id)
            )
            return result.scalar_one_or_none()
        
    async def get_by_unique_name(self, model_unique_name: str) -> KbotMdModels | None:
        """Get knowledge base model by model unique name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.model_unique_name == model_unique_name)
            )
            return result.scalar_one_or_none()
    
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
      
    async def get_by_provider(self, provider: str) -> Sequence[KbotMdModels]:
        """Get knowledge base models by provider."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.provider == provider)
            )
            return result.scalars().all()
        
    async def get_unique_name_by_id(self, model_id: int) -> str | None:
        """Get knowledge base model unique name by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels.model_unique_name).where(KbotMdModels.model_id == model_id)
            )
            return result.scalar_one_or_none()
    