from loguru import logger
from sqlalchemy import select, delete
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import PromptEntity
from .base_repo import BaseRepository


class PromptRepository(BaseRepository[PromptEntity]):
    """Repository for KBOT_MD_PROMPT table operations."""
    
    async def get_prompt_by_id(self, prompt_id: int) -> str:
        """
        Get prompt content by ID.
        Fix type annotation: return str instead of Sequence[str] (single template string)
        :param prompt_id: Prompt ID to query
        :return: Prompt template string
        """
        try:
            stmt = select(PromptEntity.template).where(PromptEntity.prompt_id == prompt_id)
            result = await self.session.execute(stmt)
            prompt_template = result.scalar_one_or_none()
            
            if prompt_template is None:
                raise DataNotFoundException(f"Prompt with ID {prompt_id} not found")
            
            logger.debug(f"Retrieved prompt template for ID {prompt_id}: {prompt_template[:50]}...")
            return prompt_template
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"Failed to get prompt by ID {prompt_id}", original_error=e)
        
    async def get_prompt_by_unique_name(self, prompt_unique_name: str) -> str:
        """
        Get prompt content by unique name.
        :param prompt_unique_name: Prompt unique name to query
        :return: Prompt template string
        """
        try:
            stmt = select(PromptEntity.template).where(
                PromptEntity.prompt_unique_name == prompt_unique_name
            )
            result = await self.session.execute(stmt)
            prompt_template = result.scalar_one_or_none()
            
            if prompt_template is None:
                raise DataNotFoundException(f"Prompt with unique name '{prompt_unique_name}' not found")
            
            logger.debug(f"Retrieved prompt template for '{prompt_unique_name}': {prompt_template[:50]}...")
            return prompt_template
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(
                f"Failed to get prompt by unique name '{prompt_unique_name}'",
                original_error=e
            )
        
    async def delete(self, prompt_id: int):
        """
        Delete prompt by ID.
        :param prompt_id: Prompt ID to delete
        """
        try:
            stmt = delete(PromptEntity).where(PromptEntity.prompt_id == prompt_id)
            await self.session.execute(stmt)

        except Exception as e:
            raise DatabaseException(f"Failed to delete prompt by ID {prompt_id}", original_error=e)
        