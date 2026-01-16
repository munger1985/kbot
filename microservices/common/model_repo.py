from typing import Sequence
from sqlalchemy import select, and_
from .model_entity import KbotMdModels
from core.dictionary import Status
from core.database.meta_oracle import get_session


class KbotMdModelsRepository:
    """Repository for KBOT_MD_KB_MODELS table operations."""
    
    def __init__(self):
        """
        初始化模型仓库
        """

    async def get_available_by_category(self, model_category: int) -> Sequence[KbotMdModels]:
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
        
    async def get_by_name(self, model_name: str) -> KbotMdModels | None:
        """Get knowledge base model by ID."""
        async with get_session() as session:
            result = await session.execute(
                select(KbotMdModels).where(KbotMdModels.model_name == model_name)
            )
            return result.scalar_one_or_none()