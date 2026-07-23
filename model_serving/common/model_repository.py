"""模型目录持久化访问。"""

from typing import Sequence
from uuid import UUID

from sqlalchemy import and_, select

from platform_core.exceptions import DatabaseException, DataNotFoundException

from .entities.ai_model import AIModelEntity
from .repository_base import ModelRepositoryBase


class AIModelRepository(ModelRepositoryBase[AIModelEntity]):
    """只提供模型目录需要的查询和状态变更。"""

    async def get_by_id(self, model_id: UUID) -> AIModelEntity:
        try:
            result = await self.session.execute(
                select(AIModelEntity).where(AIModelEntity.model_id == model_id)
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise DataNotFoundException(f"模型不存在：{model_id}")
            return model
        except DataNotFoundException:
            raise
        except Exception as exc:
            raise DatabaseException("按 ID 读取模型失败", original_error=exc)

    async def get_by_served_name(
        self, *, app_id: int, served_model_name: str,
    ) -> AIModelEntity:
        try:
            result = await self.session.execute(
                select(AIModelEntity).where(
                    AIModelEntity.app_id == app_id,
                    AIModelEntity.served_model_name == served_model_name,
                )
            )
            model = result.scalar_one_or_none()
            if model is None:
                raise DataNotFoundException(
                    f"模型不存在：{served_model_name}"
                )
            return model
        except DataNotFoundException:
            raise
        except Exception as exc:
            raise DatabaseException("按服务名读取模型失败", original_error=exc)

    async def list_by_scope(
        self, *, app_id: int, category: int | None = None,
    ) -> Sequence[AIModelEntity]:
        try:
            conditions = [AIModelEntity.app_id == app_id]
            if category is not None:
                conditions.append(AIModelEntity.category == category)
            result = await self.session.execute(
                select(AIModelEntity)
                .where(and_(*conditions))
                .order_by(AIModelEntity.served_model_name)
            )
            return result.scalars().all()
        except Exception as exc:
            raise DatabaseException("读取模型目录失败", original_error=exc)

    async def add(self, model: AIModelEntity) -> AIModelEntity:
        self.session.add(model)
        await self.session.flush()
        return model

    async def update_fields(
        self, model_id: UUID, *, app_id: int, values: dict,
    ) -> AIModelEntity:
        model = await self.get_by_id(model_id)
        if int(model.app_id) != int(app_id):
            raise DataNotFoundException(f"模型不存在：{model_id}")
        mutable_fields = {
            "display_name",
            "api_endpoint",
            "api_key",
            "status",
            "descs",
            "updated_by",
        }
        for field, value in values.items():
            if field not in mutable_fields:
                raise ValueError(f"不允许修改模型字段：{field}")
            setattr(model, field, value)
        await self.session.flush()
        return model
