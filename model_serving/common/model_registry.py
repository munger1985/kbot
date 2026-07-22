"""Model definition CRUD and lifecycle management for all model processes."""
from collections.abc import Callable
from typing import Any

from platform_core.config.settings import get_app_config, get_embed_config
from .entities.ai_model import AIModelEntity
from .model_repository import AIModelRepository


class ModelDefinitionNotFound(LookupError):
    pass


class ModelRegistryService:
    def __init__(self, *, app_id: int | None = None, session_factory: Callable):
        self._app_id = int(app_id if app_id is not None else get_app_config().app_id)
        self._session_factory = session_factory

    @staticmethod
    def _safe(entity: AIModelEntity) -> dict[str, Any]:
        return {
            "model_id": int(entity.model_id), "app_id": int(entity.app_id),
            "display_name": entity.display_name, "model_name": entity.model_name,
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

    async def get(self, model_id: int, *, category: int | None = None) -> dict[str, Any]:
        async with self._session_factory() as session:
            row = await AIModelRepository(session).get_by_id(model_id)
            if int(row.app_id) != self._app_id or (category is not None and int(row.category or 0) != int(category)):
                raise ModelDefinitionNotFound(model_id)
            return self._safe(row)

    async def create(self, values: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
        self._validate_embedding_dimension(values)
        async with self._session_factory() as session:
            row = await AIModelRepository(session).add(AIModelEntity(
                app_id=self._app_id, display_name=values["display_name"], model_name=values["model_name"],
                category=values["category"], provider=values["provider"], api_endpoint=values.get("api_endpoint"),
                api_key=values.get("api_key"), status=values.get("status", 0),
                embedding_dimension=values.get("embedding_dimension"), model_params=values.get("model_params") or {},
                descs=values.get("descs"), created_by=actor_id, updated_by=actor_id,
            ))
            await session.commit()
            return self._safe(row)

    async def update(self, model_id: int, values: dict[str, Any], *, actor_id: str) -> dict[str, Any]:
        async with self._session_factory() as session:
            repo = AIModelRepository(session)
            current = await repo.get_by_id(model_id)
            if int(current.app_id) != self._app_id:
                raise ModelDefinitionNotFound(model_id)
            check_values = {"category": current.category, **values}
            self._validate_embedding_dimension(check_values)
            row = await repo.update_fields(model_id, app_id=self._app_id, values={**values, "updated_by": actor_id})
            await session.commit()
            return self._safe(row)

    @staticmethod
    def _validate_embedding_dimension(values: dict[str, Any]) -> None:
        # The application-wide vector dimension is a hard invariant.  A model
        # definition may omit it only for non-embedding categories.
        if int(values.get("category", 0) or 0) != 2 or values.get("embedding_dimension") is None:
            return
        configured = get_embed_config().dimensions
        if configured is not None and int(values["embedding_dimension"]) != int(configured):
            raise ValueError(
                f"embedding_dimension must equal the configured vector dimension {configured}"
            )

    async def delete(self, model_id: int, *, actor_id: str) -> None:
        async with self._session_factory() as session:
            await AIModelRepository(session).update_fields(model_id, app_id=self._app_id, values={"status": 2, "updated_by": actor_id})
            await session.commit()
