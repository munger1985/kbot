"""模型目录、生命周期、引用检查与缓存失效。"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import StaleDataError
from loguru import logger

from model_serving.common.entities.ai_model import AIModelEntity
from model_serving.common.provider_catalog import validate_provider_config
from model_serving.config import get_model_serving_settings
from platform_core.contracts import AuthContext, ModelReferenceSummary
from platform_core.exceptions import DataNotFoundException
from platform_core.identity import uuid7


_STATUS_TO_DB = {"DRAFT": 0, "ACTIVE": 1, "ARCHIVED": 2}
_STATUS_FROM_DB = {value: key for key, value in _STATUS_TO_DB.items()}


class ModelRegistryError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelDefinitionNotFound(ModelRegistryError):
    def __init__(self, model_id: UUID):
        super().__init__("MODEL_NOT_FOUND", f"模型不存在：{model_id}")


class ModelRegistryConflict(ModelRegistryError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message)
        self.details = details or {}


ReferenceResolver = Callable[[UUID, AuthContext], Awaitable[list[dict[str, Any]]]]
InvalidationHandler = Callable[[dict[str, Any]], Awaitable[None]]


class ModelRegistryService:
    """通过 UoW 管理目录，Repository 和路由都不拥有事务。"""

    def __init__(
        self,
        *,
        uow_factory,
        on_model_changed: InvalidationHandler | None = None,
        reference_resolvers: Mapping[str, ReferenceResolver] | None = None,
        is_model_loaded: Callable[[str], bool] | None = None,
        notification_publisher=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._on_model_changed = on_model_changed
        self._reference_resolvers = dict(reference_resolvers or {})
        self._is_model_loaded = is_model_loaded
        self._notification_publisher = notification_publisher

    @staticmethod
    def _safe(entity: AIModelEntity) -> dict[str, Any]:
        """目录投影不得包含 API Key 等连接密钥。"""
        return {
            "model_id": str(entity.model_id),
            "served_model_name": entity.served_model_name,
            "display_name": entity.display_name,
            "provider_model_name": entity.provider_model_name,
            "category": int(entity.category),
            "provider": entity.provider,
            "api_endpoint": entity.api_endpoint,
            "status": _STATUS_FROM_DB[int(entity.status)],
            "model_params": entity.model_params or {},
            "description": entity.descs,
            "row_version": int(entity.row_version),
        }

    async def list(self, *, category: int | None = None) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.models
            rows = await uow.models.list_by_scope(category=category)
            return [self._safe(row) for row in rows]

    async def get(
        self, model_id: UUID, *, category: int | None = None,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.models
            try:
                row = await uow.models.get_by_id(model_id)
            except DataNotFoundException as exc:
                raise ModelDefinitionNotFound(model_id) from exc
            if category is not None and int(row.category) != category:
                raise ModelDefinitionNotFound(model_id)
            return self._safe(row)

    async def create(
        self, values: dict[str, Any], *, actor_id: str,
        auth_context: AuthContext | None = None,
    ) -> dict[str, Any]:
        values = dict(values)
        self._validate(values)
        status = str(values.pop("status", "DRAFT"))
        async with self._uow_factory() as uow:
            assert uow.models
            row = AIModelEntity(
                model_id=uuid7(),
                served_model_name=values["served_model_name"],
                display_name=values["display_name"],
                provider_model_name=values["provider_model_name"],
                category=values["category"],
                provider=values["provider"],
                api_endpoint=values.get("api_endpoint"),
                api_key=values.get("api_key"),
                status=_STATUS_TO_DB[status],
                model_params=values.get("model_params") or {},
                descs=values.get("description"),
                created_by=actor_id,
                updated_by=actor_id,
            )
            try:
                await uow.models.add(row)
                await uow.commit()
            except IntegrityError as exc:
                raise ModelRegistryConflict(
                    "MODEL_NATURAL_KEY_CONFLICT", "模型服务名已经存在",
                ) from exc
            result = self._safe(row)
        await self._notify_changed(result, auth_context=auth_context)
        return result

    async def update(
        self, model_id: UUID, values: dict[str, Any], *,
        expected_row_version: int, actor_id: str,
        auth_context: AuthContext | None = None,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.models
            row = await self._locked(uow.models, model_id)
            self._assert_version(row, expected_row_version)
            if int(row.status) == _STATUS_TO_DB["ARCHIVED"]:
                raise ModelRegistryConflict(
                    "MODEL_ARCHIVED", "归档模型不能继续修改",
                )
            effective = {
                "category": int(row.category), "provider": row.provider,
                "api_endpoint": row.api_endpoint, "api_key": row.api_key,
                "model_params": row.model_params or {}, **values,
            }
            self._assert_embedding_space_unchanged(row, effective)
            self._validate(effective)
            mutable = {
                "display_name": "display_name",
                "api_endpoint": "api_endpoint",
                "api_key": "api_key",
                "model_params": "model_params",
                "description": "descs",
            }
            for source, target in mutable.items():
                if source in values:
                    setattr(row, target, values[source])
            row.updated_by = actor_id
            try:
                await uow.flush()
                await uow.commit()
            except StaleDataError as exc:
                raise ModelRegistryConflict(
                    "MODEL_VERSION_CONFLICT", "模型目录版本已变化",
                ) from exc
            result = self._safe(row)
        await self._notify_changed(result, auth_context=auth_context)
        return result

    async def change_status(
        self, model_id: UUID, *, target_status: str,
        expected_row_version: int, actor_id: str,
        auth_context: AuthContext | None = None,
    ) -> dict[str, Any]:
        if target_status not in {"DRAFT", "ACTIVE"}:
            raise ValueError("状态接口只接受 DRAFT 或 ACTIVE")
        async with self._uow_factory() as uow:
            assert uow.models
            row = await self._locked(uow.models, model_id)
            self._assert_version(row, expected_row_version)
            if int(row.status) == _STATUS_TO_DB["ARCHIVED"]:
                raise ModelRegistryConflict(
                    "MODEL_ARCHIVED", "归档状态不可逆",
                )
            row.status = _STATUS_TO_DB[target_status]
            row.updated_by = actor_id
            await uow.flush()
            await uow.commit()
            result = self._safe(row)
        await self._notify_changed(result, auth_context=auth_context)
        return result

    async def archive(
        self, model_id: UUID, *, expected_row_version: int, actor_id: str,
        auth_context: AuthContext,
    ) -> tuple[dict[str, Any], ModelReferenceSummary]:
        references = await self.references(model_id, auth_context=auth_context)
        if references.unavailable_services or references.references:
            await self._publish_blocked(
                event_type="model.catalog.archive_blocked",
                model_id=model_id, auth_context=auth_context,
                references=references,
            )
            raise ModelRegistryConflict(
                "MODEL_REFERENCED",
                "模型仍被引用或引用检查服务不可用，不能归档",
                details=references.model_dump(mode="json"),
            )
        async with self._uow_factory() as uow:
            assert uow.models
            row = await self._locked(uow.models, model_id)
            self._assert_version(row, expected_row_version)
            row.status = _STATUS_TO_DB["ARCHIVED"]
            row.updated_by = actor_id
            await uow.flush()
            await uow.commit()
            result = self._safe(row)
        await self._notify_changed(result, auth_context=auth_context)
        return result, references

    async def delete(
        self, model_id: UUID, *, expected_row_version: int,
        auth_context: AuthContext,
    ) -> None:
        references = await self.references(model_id, auth_context=auth_context)
        if references.unavailable_services:
            await self._publish_blocked(
                event_type="model.catalog.delete_blocked",
                model_id=model_id, auth_context=auth_context,
                references=references,
            )
            raise ModelRegistryConflict(
                "MODEL_REFERENCE_CHECK_UNAVAILABLE",
                "引用检查服务不可用：" + ", ".join(references.unavailable_services),
                details=references.model_dump(mode="json"),
            )
        if references.references:
            await self._publish_blocked(
                event_type="model.catalog.delete_blocked",
                model_id=model_id, auth_context=auth_context,
                references=references,
            )
            raise ModelRegistryConflict(
                "MODEL_REFERENCED", "模型仍被业务配置或运行记录引用",
                details=references.model_dump(mode="json"),
            )
        async with self._uow_factory() as uow:
            assert uow.models
            row = await self._locked(uow.models, model_id)
            self._assert_version(row, expected_row_version)
            if int(row.status) != _STATUS_TO_DB["ARCHIVED"]:
                raise ModelRegistryConflict(
                    "MODEL_NOT_ARCHIVED", "只有归档模型可以删除",
                )
            if self._is_model_loaded and self._is_model_loaded(row.served_model_name):
                raise ModelRegistryConflict(
                    "MODEL_INSTANCE_RUNNING", "模型实例仍在运行",
                )
            event = self._safe(row)
            await uow.models.delete(row)
            await uow.commit()
        await self._notify_changed(event, auth_context=auth_context)

    async def references(
        self, model_id: UUID, *, auth_context: AuthContext,
    ) -> ModelReferenceSummary:
        references: list[dict[str, Any]] = []
        unavailable: list[str] = []
        for service, resolver in self._reference_resolvers.items():
            try:
                references.extend(await resolver(model_id, auth_context))
            except Exception:
                unavailable.append(service)
        return ModelReferenceSummary(
            model_id=model_id,
            references=tuple(references),
            unavailable_services=tuple(unavailable),
        )

    @staticmethod
    async def _locked(repository, model_id: UUID) -> AIModelEntity:
        try:
            return await repository.get_by_id(model_id, lock=True)
        except DataNotFoundException as exc:
            raise ModelDefinitionNotFound(model_id) from exc

    @staticmethod
    def _assert_version(row: AIModelEntity, expected: int) -> None:
        if int(row.row_version) != expected:
            raise ModelRegistryConflict(
                "MODEL_VERSION_CONFLICT", "模型目录版本已变化",
            )

    @staticmethod
    def _assert_embedding_space_unchanged(
        row: AIModelEntity, effective: dict[str, Any],
    ) -> None:
        if int(row.category) != 2:
            return
        current = (row.model_params or {}).get("embedding_dimension")
        updated = (effective.get("model_params") or {}).get("embedding_dimension")
        if current != updated:
            raise ModelRegistryConflict(
                "MODEL_VECTOR_SPACE_IMMUTABLE",
                "Embedding 维度不可原地修改，请创建新模型",
            )

    @staticmethod
    def _validate(values: dict[str, Any]) -> None:
        validate_provider_config(values)
        category = int(values.get("category") or 0)
        params = values.get("model_params") or {}
        dimension = params.get("embedding_dimension")
        if category != 2:
            if dimension is not None:
                raise ValueError("非文本 Embedding 模型不能设置 embedding_dimension")
            return
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("model_params.embedding_dimension 必须为正整数")
        configured = get_model_serving_settings().vector.dimensions
        if int(dimension) != int(configured):
            raise ValueError(f"embedding_dimension 必须等于配置维度 {configured}")

    async def _notify_changed(
        self, model: dict[str, Any], *, auth_context: AuthContext | None,
    ) -> None:
        if self._on_model_changed is None:
            return
        try:
            await self._on_model_changed({
                "model_id": model["model_id"],
                "served_model_name": model["served_model_name"],
                "category": model["category"],
                "row_version": model["row_version"],
            })
        except Exception as exc:
            if self._notification_publisher is not None and auth_context is not None:
                try:
                    await self._notification_publisher.publish_reload_failed(
                        model=model,
                        auth_context=auth_context,
                        error_code=type(exc).__name__.upper(),
                    )
                except Exception as publish_exc:
                    logger.error(
                        "模型运行时重载失败通知写入异常 | model_id={} | error_type={}",
                        model["model_id"], type(publish_exc).__name__,
                    )
            raise

    async def _publish_blocked(
        self, *, event_type: str, model_id: UUID,
        auth_context: AuthContext, references: ModelReferenceSummary,
    ) -> None:
        if self._notification_publisher is None:
            return
        try:
            model = await self.get(model_id)
            model_name = str(model["display_name"])
        except ModelRegistryError:
            model_name = str(model_id)
        await self._notification_publisher.publish_blocked(
            event_type=event_type, model_id=model_id,
            model_name=model_name, auth_context=auth_context,
            references=references,
        )
