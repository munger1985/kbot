from typing import Sequence
from loguru import logger
from sqlalchemy import select, and_, update
from core.exceptions import DatabaseException, DataNotFoundException
from core.dictionary import Status
from dao.entities import AIModelEntity
from .base_repo import BaseRepository
from core.config.settings import get_app_config


class AIModelRepository(BaseRepository[AIModelEntity]):
    """Repository for KBOT_MD_KB_MODELS table operations."""

    async def toggle_model(self, model_id: int, status: Status):
        """
        Toggle model status by ID.
        :param model_id: Model ID to enable
        """
        try:
            # 验证模型存在
            await self.get_by_id(model_id)
            
            stmt = update(AIModelEntity).where(
                AIModelEntity.model_id == model_id
            ).values(status=status.value).returning(AIModelEntity.model_id)
            
            result = await self.session.execute(stmt)
            
            if not result.scalar():
                raise DatabaseException(f"Failed to {'enable' if status == Status.ENABLED else 'disable'} model with ID {model_id}")
            logger.info(f"{'Enabled' if status == Status.ENABLED else 'Disabled'} model with ID: {model_id}")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to toggle model", original_error=e)

    async def get_category_by_id(self, model_id: int) -> int:
        """
        Get the category of a model by its ID.
        :param model_id: Model ID to query
        :return: Model category integer
        """
        try:
            stmt = select(AIModelEntity.category).where(AIModelEntity.model_id == model_id)
            result = await self.session.execute(stmt)
            category = result.scalar_one_or_none()
            
            if category is None:
                raise DataNotFoundException(f"Category not found for model ID {model_id}")
            
            return category
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model category by ID", original_error=e)
    
    async def get_display_name_by_id(self, model_id: int) -> str:
        """
        Get knowledge base model display name by ID.
        :param model_id: Model ID to query
        :return: Model display name string
        """
        try:
            stmt = select(AIModelEntity.display_name).where(AIModelEntity.model_id == model_id)
            result = await self.session.execute(stmt)
            display_name = result.scalar_one_or_none()
            
            if display_name is None:
                raise DataNotFoundException(f"Display name not found for model ID {model_id}")
            
            return display_name
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model display name by ID", original_error=e)

    async def get_available_by_category(self, model_category: int) -> Sequence[AIModelEntity]:
        """
        Get all available (enabled) models by category.
        :param model_category: Model category to filter
        :return: Sequence of AIModelEntity instances
        """
        try:
            stmt = select(AIModelEntity).where(
                and_(
                    AIModelEntity.category == model_category,
                    AIModelEntity.status == Status.ENABLED.value
                )
            )
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            raise DatabaseException("Failed to get available models by category", original_error=e)
        
    async def get_by_id(self, model_id: int) -> AIModelEntity:
        """
        Get knowledge base model by ID.
        :param model_id: Model ID to query
        :return: AIModelEntity instance
        """
        try:
            stmt = select(AIModelEntity).where(AIModelEntity.model_id == model_id)
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                raise DataNotFoundException(f"Model with ID {model_id} not found")
            
            return model
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model by ID", original_error=e)
        
    async def get_name_by_id(self, model_id: int) -> str:
        """
        Get knowledge base model name by ID.
        :param model_id: Model ID to query
        :return: Model name string
        """
        try:
            stmt = select(AIModelEntity.model_name).where(AIModelEntity.model_id == model_id)
            result = await self.session.execute(stmt)
            model_name = result.scalar_one_or_none()
            
            if model_name is None:
                raise DataNotFoundException(f"Model name not found for ID {model_id}")
            
            return model_name
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model name by ID", original_error=e)
        
    async def get_by_name(self, model_name: str) -> AIModelEntity:
        """
        Get knowledge base model by name.
        :param model_name: model name
        :return: AIModelEntity
        """
        try:
            stmt = select(AIModelEntity).where(
                and_(
                    AIModelEntity.display_name == model_name,
                    AIModelEntity.app_id == get_app_config().app_id
                )
            )
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                raise DataNotFoundException(f"Model with name '{model_name}' not found")
            
            return model
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"Failed to get model by name '{model_name}'", original_error=e)
        
    async def get_by_display_name(self, display_name: str) -> AIModelEntity:
        """
        Get knowledge base model by display name.
        :param display_name: Model display name to query
        :return: AIModelEntity
        """
        try:
            stmt = select(AIModelEntity).where(
                and_(
                    AIModelEntity.display_name == display_name),
                    AIModelEntity.app_id == get_app_config().app_id
                )
            result = await self.session.execute(stmt)
            model = result.scalar_one_or_none()
            
            if not model:
                raise DataNotFoundException(f"Model with display name '{display_name}' not found")
            
            return model
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"Failed to get model by display name '{display_name}'", original_error=e)