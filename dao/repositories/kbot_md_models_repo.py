import json
from typing import Sequence
from sqlalchemy import select, and_
from dao.entities.kbot_md_models import KbotMdModels
from core.dictionary import Status
from core.database.meta_oracle import get_session
from core.database.meta_redis import AsyncRedisPool
from utils.decimal_encoder import DecimalEncoder


class KbotMdModelsRepository:
    """Repository for KBOT_MD_KB_MODELS table operations."""
    
    def __init__(self):
        """
        初始化模型仓库，内部创建 AsyncRedisPool 实例
        """
        self.redis = AsyncRedisPool(db=1) # 1 为模型数据库
    
    async def sync_all_available_models(self) -> bool:
        """获取所有可用模型并同步到 redis """
        async with get_session() as session:
            query = select(KbotMdModels).where(
                KbotMdModels.status == Status.ENABLED.value  # 只获取启用状态的模型
            )
            result = await session.execute(query)
            models = result.scalars().all()
            if not models:
                return False

            async with self.redis as redis:
                for model in models:
                    model_id = int(model.model_id)
                    category = int(model.category) if model.category else 0
                    model_params = json.dumps(model.model_params, cls=DecimalEncoder) if model.model_params else {}
                    model_data = {
                        "model_id": model_id,
                        "model_unique_name": model.model_unique_name,
                        "model_name": model.model_name,
                        "category": category,
                        "provider": model.provider,
                        "api_endpoint": model.api_endpoint if model.api_endpoint else "",
                        "api_key": model.api_key if model.api_key else "",
                        "model_params": model_params
                    }

                    # 直接写入 Redis
                    await redis.hset(f"model:{model_id}", mapping=model_data)
                    await redis.set(f"index:unique_name:{model.model_unique_name}", model_id)
                    await redis.sadd(f"index:category:{category}", model.model_unique_name)
                
            return True



    async def get_all_models_by_category(self, model_category: int) -> Sequence[KbotMdModels]:
        """获取所有指定类型的可用模型"""
        async with get_session() as session:
            query = select(KbotMdModels).where(
                and_(
                    KbotMdModels.category == model_category,
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
            # 将模型数据同步到 redis
            async with self.redis as redis:
                model_id = int(model.model_id)
                category = int(model.category) if model.category else 0
                model_params = json.dumps(model.model_params, cls=DecimalEncoder) if model.model_params else {}
                model_data = {
                    "model_id": model_id,
                    "model_unique_name": model.model_unique_name,
                    "model_name": model.model_name,
                    "category": category,
                    "provider": model.provider,
                    "api_endpoint": model.api_endpoint if model.api_endpoint else "",
                    "api_key": model.api_key if model.api_key else "",
                    "model_params": model_params
                }

                # 直接写入 Redis
                await redis.hset(f"model:{model_id}", mapping=model_data)
                await redis.set(f"index:unique_name:{model.model_unique_name}", model_id)
                await redis.sadd(f"index:category:{category}", model_id)

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
            # 使用 merge 来更新现有记录或插入新记录
            merged_model = await session.merge(model)
            await session.commit()
            await session.refresh(merged_model)

            # 将模型数据同步到 redis
            async with self.redis as redis:
                model_id = int(model.model_id)
                category = int(model.category) if model.category else 0
                model_params = json.dumps(model.model_params, cls=DecimalEncoder) if model.model_params else {}
                model_data = {
                    "model_id": model_id,
                    "model_unique_name": model.model_unique_name,
                    "model_name": model.model_name,
                    "category": category,
                    "provider": model.provider,
                    "api_endpoint": model.api_endpoint if model.api_endpoint else "",
                    "api_key": model.api_key if model.api_key else "",
                    "model_params": model_params
                }

                # 直接写入 Redis
                await redis.hset(f"model:{model_id}", mapping=model_data)
                await redis.set(f"index:unique_name:{model.model_unique_name}", model_id)
                await redis.sadd(f"index:category:{category}", model_id)

            return merged_model
    
    async def delete(self, model_id: int) -> bool:
        """Delete a knowledge base model record by ID."""
        async with get_session() as session:
            # 更高效的查询方式，直接使用 session.get()
            model = await session.get(KbotMdModels, model_id)
            if model is None:
                return False
            
            await session.delete(model)
            await session.commit()
            await self.redis.delete(f"model:{model_id}")
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
    
    async def get_provider_by_unique_name(self, model_unique_name: str) -> str | None:
        """Get knowledge base model provider by unique name."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels.provider).where(
                    KbotMdModels.model_unique_name == model_unique_name
                )
            )
            return result.scalar_one_or_none()