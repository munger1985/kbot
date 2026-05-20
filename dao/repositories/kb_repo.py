from typing import Sequence
from loguru import logger
from sqlalchemy import select, delete, update, and_
from core.exceptions import DatabaseException, DataNotFoundException
from dao.entities import KBEntity, AgentConfEntity
from .base_repo import BaseRepository


class KBRepository(BaseRepository[KBEntity]):
    """Repository for KBOT_MD_KB table operations."""
    
    async def create(self, kb: KBEntity) -> None:
        """创建知识库"""
        try:
            self.session.add(kb)
        except Exception as e:
            raise DatabaseException(f"创建知识库失败", original_error=e)

    async def update(self, kb_id: int, **kwargs) -> None:
        """更新知识库"""
        try:
            result = await self.session.execute(
                update(KBEntity)
                .where(KBEntity.kb_id == kb_id)
                .values(**kwargs)
                .returning(KBEntity)
            )
            
            if result.scalar() is None:
                raise DataNotFoundException(f"知识库 {kb_id} 不存在")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"更新知识库失败", original_error=e)

    async def delete(self, kb_id: int) -> None:
        """根据ID删除知识库"""
        try:
            await self.session.execute(
                delete(KBEntity)
                .where(KBEntity.kb_id == kb_id)
            )
        except Exception as e:
            raise DatabaseException(f"根据ID删除知识库失败", original_error=e)

    async def get_all(self, domain_id: int | None = None, 
                      category: int | None = None,
                      is_active: bool | None = None) -> Sequence[KBEntity]:
        """获取所有知识库"""
        try:
            query = select(KBEntity)
            if domain_id:
                query = query.where(KBEntity.domain_id == domain_id)
            if is_active is not None:
                query = query.where(KBEntity.kb_status == 1 if is_active else KBEntity.kb_status != 1)
            if category is not None:
                query = query.where(KBEntity.kb_category == category)
                
            result = await self.session.execute(query)
            kbs = result.scalars().all()
            if not kbs:
                raise DataNotFoundException(f"知识库不存在")
            return kbs
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"获取所有知识库失败", original_error=e)

    async def toggle_active(self, kb_id: int, is_active: bool, user_name: str) -> None:
        """切换知识库状态"""
        try:
            result = await self.session.execute(
                update(KBEntity)
                .where(KBEntity.kb_id == kb_id)
                .values(is_active=is_active, updated_by=user_name)
                .returning(KBEntity.kb_id)
            )
            
            if result.scalar() is None:
                raise DataNotFoundException(f"知识库 {kb_id} 不存在")
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"切换知识库状态失败", original_error=e)

    async def get_name_by_id(self, kb_id: int) -> str:
        """根据ID获取知识库名称"""
        try:
            result = await self.session.execute(
                select(KBEntity.kb_name)
                .where(KBEntity.kb_id == kb_id)
            )
            kb_name = result.scalar_one_or_none()
            if not kb_name:
                raise DataNotFoundException(f"知识库 {kb_id} 名称不存在")
            return kb_name
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException(f"根据ID获取知识库名称失败", original_error=e)

    
    async def get_by_agent_and_category(self, agent_id: str, category: int) -> list[dict[str, str]]:
        """根据 Agent ID 和知识库类型获取所有配置"""
        try:
            stmt = (
                select(KBEntity.kb_id, KBEntity.kb_name, KBEntity.descs)
                .select_from(AgentConfEntity)  # 明确以关联表为起点
                .join(KBEntity, AgentConfEntity.kb_id == KBEntity.kb_id)
                .where(
                    and_(
                        AgentConfEntity.agent_id == agent_id, 
                        KBEntity.kb_category == category
                    )
                )
            )
            
            result = await self.session.execute(stmt)
            rows = result.mappings().all()
            
            if not rows:
                raise DataNotFoundException(f"未找到知识库配置 Agent:{agent_id} Category:{category}")
                
            return [
                {
                    "id": str(row.kb_id),
                    "name": row.kb_name,
                    "descs": row.descs or ""
                } for row in rows
            ]

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("获取知识库配置失败", original_error=e)
        
    async def if_exists_in_domain(self, domain_id: int) -> bool:
        """检查知识库是否在指定业务域中"""
        try:
            result = await self.session.execute(
                select(1).where(KBEntity.domain_id == domain_id)
            )
            return result.scalar_one_or_none() is not None
        except Exception as e:
            raise DatabaseException(f"检查知识库是否在指定业务域中失败", original_error=e)
        
    async def get_by_id(self, kb_id: int) -> KBEntity:
        """
        Get knowledge base by ID.
        :param kb_id: Knowledge base ID
        :return: KBEntity instance
        """
        try:
            stmt = select(KBEntity).where(KBEntity.kb_id == kb_id)
            result = await self.session.execute(stmt)
            kb = result.scalar_one_or_none()
            
            if not kb:
                raise DataNotFoundException(f"Knowledge base with ID {kb_id} not found")
            
            return kb
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base by ID", original_error=e)
    
    async def get_by_name(self, kb_name: str) -> KBEntity:
        """
        Get knowledge base by name.
        :param kb_name: Knowledge base name
        :return: KBEntity instance
        """
        try:
            stmt = select(KBEntity).where(KBEntity.kb_name == kb_name)
            result = await self.session.execute(stmt)
            kb = result.scalars().first()
            
            if not kb:
                raise DataNotFoundException(f"Knowledge base '{kb_name}' not found")
            
            return kb
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get knowledge base by name", original_error=e)
    
    async def get_by_app_domain_name(self, app_id: int, domain_id: int, kb_name: str) -> KBEntity:
        """
        Get knowledge base by app_id, domain_id and kb_name (unique constraint).
        :param app_id: Application ID
        :param domain_id: Domain ID
        :param kb_name: Knowledge base name
        :return: KBEntity instance
        """
        try:
            stmt = select(KBEntity).where(
                KBEntity.app_id == app_id,
                KBEntity.domain_id == domain_id,
                KBEntity.kb_name == kb_name
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
        :return: Dict containing model configuration from the models JSON field
        """
        try:
            stmt = select(
                KBEntity.kb_category,
                KBEntity.models
            ).where(KBEntity.kb_id == kbid)
            
            result = await self.session.execute(stmt)
            model_config = result.fetchone()
            
            if not model_config:
                raise DataNotFoundException(f"Model configuration not found for KB ID {kbid}")
            
            kb_category, models = model_config
            models_dict = models if models else {}
            
            return {
                "kb_category": kb_category,
                "img2txt_model_id": models_dict.get("img2txt_model_id"),
                "img_embed_model_id": models_dict.get("img_embed_model_id"),
                "txt_embed_model_id": models_dict.get("txt_embed_model_id")
            }
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            raise DatabaseException("Failed to get model configuration by KB ID", original_error=e)