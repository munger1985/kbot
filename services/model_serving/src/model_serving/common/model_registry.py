"""所有模型进程共享的模型定义与生命周期管理。"""
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from model_serving.config import get_model_serving_settings
from .entities.ai_model import AIModelEntity
from .model_repository import AIModelRepository


class ModelDefinitionNotFound(LookupError):
    pass


class ModelRegistryService:
    def __init__(
        self, *,
        app_id: int | None = None,
        session_factory: Callable,
        on_model_changed: Callable[[str], Awaitable[None]] | None = None,
    ):
        self._app_id = int(
            app_id
            if app_id is not None
            else get_model_serving_settings().platform.app_id
        )
        self._session_factory = session_factory
        self._on_model_changed = on_model_changed

    @staticmethod
    def _safe(entity: AIModelEntity) -> dict[str, Any]:
        return {
            "model_id": str(entity.model_id), "app_id": int(entity.app_id),
            "served_model_name": entity.served_model_name,
            "display_name": entity.display_name,
            "provider_model_name": entity.provider_model_name,
            "category": int(entity.category) if entity.category is not None else None,
            "provider": entity.provider, "api_endpoint": entity.api_endpoint,
            "status": int(entity.status) if entity.status is not None else None,
            "embedding_dimension": int(entity.embedding_dimension) if entity.embedding_dimension is not None else None,
            "model_params": entity.model_params or {}, "descs": entity.descs,
            "created_by": entity.created_by, "updated_by": entity.updated_by,
        }

    async def list(self, *, category: int | None = None) -> list[dict[str, Any]]:
        async with self._session_factory() as session:
            rows = await AIModelRepository(session).list_by_scope(app_id=self._app_id, category=category)
            return [self._safe(row) for row in rows]

    async def get(self, model_id: UUID, *, category: int | None = None) -> dict[str, Any]:
        async with self._session_factory() as session:
            row = await AIModelRepository(session).get_by_id(model_id)
            if int(row.app_id) != self._app_id or (category is not None and int(row.category or 0) != int(category)):
                raise ModelDefinitionNotFound(model_id)
            return self._safe(row)

    async def create(self, values: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
        self._validate_embedding_dimension(values)
        async with self._session_factory() as session:
            row = await AIModelRepository(session).add(AIModelEntity(
                app_id=self._app_id,
                served_model_name=values["served_model_name"],
                display_name=values["display_name"],
                provider_model_name=values["provider_model_name"],
                category=values["category"], provider=values["provider"], api_endpoint=values.get("api_endpoint"),
                api_key=values.get("api_key"), status=values.get("status", 0),
                embedding_dimension=values.get("embedding_dimension"), model_params=values.get("model_params") or {},
                descs=values.get("descs"), created_by=actor_id, updated_by=actor_id,
            ))
            await session.commit()
            return self._safe(row)

    async def update(self, model_id: UUID, values: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
        async with self._session_factory() as session:
            repo = AIModelRepository(session)
            current = await repo.get_by_id(model_id)
            if int(current.app_id) != self._app_id:
                raise ModelDefinitionNotFound(model_id)
            check_values = {
                "category": current.category,
                "embedding_dimension": current.embedding_dimension,
                **values,
            }
            self._validate_embedding_dimension(check_values)
            row = await repo.update_fields(model_id, app_id=self._app_id, values={**values, "updated_by": actor_id})
            await session.commit()
            result = self._safe(row)
        await self._notify_changed(result["served_model_name"])
        return result

    @staticmethod
    def _validate_embedding_dimension(values: dict[str, Any]) -> None:
        """强制模型目录与物理向量列使用同一维度。"""
        category = int(values.get("category", 0) or 0)
        dimension = values.get("embedding_dimension")
        if category != 2:
            if dimension is not None:
                raise ValueError("非文本 Embedding 模型不能设置 embedding_dimension")
            return
        if dimension is None:
            raise ValueError("文本 Embedding 模型必须设置 embedding_dimension")
        configured = get_model_serving_settings().vector.dimensions
        if configured is not None and int(dimension) != int(configured):
            raise ValueError(
                f"embedding_dimension 必须等于配置维度 {configured}"
            )

    async def delete(self, model_id: UUID, *, actor_id: str) -> None:
        async with self._session_factory() as session:
            row = await AIModelRepository(session).update_fields(
                model_id,
                app_id=self._app_id,
                values={"status": 2, "updated_by": actor_id},
            )
            await session.commit()
            served_model_name = row.served_model_name
        await self._notify_changed(served_model_name)

    async def _notify_changed(self, served_model_name: str) -> None:
        if self._on_model_changed is not None:
            await self._on_model_changed(served_model_name)
