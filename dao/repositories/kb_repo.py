from typing import Sequence, Optional, Tuple, Any
from loguru import logger
from sqlalchemy import select, delete, update
from core.exceptions import DatabaseException, DataNotFoundException
from core.dictionary import KbCategory, KbStatus, Status
from dao.entities import KbEntity
from .base_repo import BaseRepository


class KBRepository(BaseRepository[KbEntity]):
    """Repository for KBOT_MD_KB table operations."""
    
    
    async def get_by_id(self, kb_id: int) -> KbEntity:
        """
        Get knowledge base by ID.
        :param kb_id: Knowledge base ID
        :return: KbEntity instance
        """
        try:
            stmt = select(KbEntity).where(KbEntity.kb_id == kb_id)
            result = await self.session.execute(stmt)
            kb = result.scalar_one_or_none()
            
            if not kb:
                raise DataNotFoundException(f"Knowledge base with ID {kb_id} not found")
            
            return kb
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base by ID", original_error=e)
    
    async def get_by_name(self, kb_name: str) -> KbEntity:
        """
        Get knowledge base by name.
        :param kb_name: Knowledge base name
        :return: KbEntity instance
        """
        try:
            stmt = select(KbEntity).where(KbEntity.kb_name == kb_name)
            result = await self.session.execute(stmt)
            kb = result.scalars().first()
            
            if not kb:
                raise DataNotFoundException(f"Knowledge base '{kb_name}' not found")
            
            return kb
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base by name", original_error=e)
    
    async def get_by_app_domain_name(self, app_id: int, domain_id: int, kb_name: str) -> KbEntity:
        """
        Get knowledge base by app_id, domain_id and kb_name (unique constraint).
        :param app_id: Application ID
        :param domain_id: Domain ID
        :param kb_name: Knowledge base name
        :return: KbEntity instance
        """
        try:
            stmt = select(KbEntity).where(
                KbEntity.app_id == app_id,
                KbEntity.domain_id == domain_id,
                KbEntity.kb_name == kb_name
            )
            result = await self.session.execute(stmt)
            kb = result.scalars().first()
            
            if not kb:
                raise DataNotFoundException(
                    f"Knowledge base '{kb_name}' not found (app_id={app_id}, domain_id={domain_id})"
                )
            
            return kb
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base by app/domain/name", original_error=e)
    
    async def get_model_by_id(self, kbid: int) -> dict[str, int | None]:
        """
        Get model configuration by knowledge base ID.
        :param kbid: Knowledge base ID
        :return: Tuple containing (kb_category, img2txt_model_id, img_embed_model_id, txt_embed_model_id, summary_model_id)
        """
        try:
            stmt = select(
                KbEntity.img2txt_model_id,
                KbEntity.img_embed_model_id,
                KbEntity.txt_embed_model_id
            ).where(KbEntity.kb_id == kbid)
            
            result = await self.session.execute(stmt)
            model_config = result.fetchone()
            
            if not model_config:
                raise DataNotFoundException(f"Model configuration not found for KB ID {kbid}")
            
            return {
                "img2txt_model_id": model_config[0],
                "img_embed_model_id": model_config[1],
                "txt_embed_model_id": model_config[2]
            }
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model configuration by KB ID", original_error=e)