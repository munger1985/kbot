"""多媒体创作工作台 Domain 模型角色绑定。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from media_studio_app.application.catalog import (
    is_role_ready, load_catalog_model, safe_model_snapshot,
)
from media_studio_app.application.errors import MediaStudioApplicationError
from media_studio_app.entities import MediaStudioModelBindingEntity
from platform_core.identity import uuid7


BindingRole = Literal["IMAGE_GENERATION"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UpsertBindingCommand(_Model):
    domain_id: int = Field(ge=1)
    role: BindingRole
    model_id: UUID
    expected_row_version: int | None = Field(default=None, ge=1)
    actor_id: str = Field(min_length=1, max_length=256)


def _ready(role: str, snapshot: dict[str, Any]) -> bool:
    return is_role_ready(role, snapshot)


def binding_view(row: MediaStudioModelBindingEntity) -> dict[str, Any]:
    snapshot = dict(row.model_snapshot_json or {})
    return {
        "binding_id": str(row.binding_id),
        "domain_id": int(row.domain_id),
        "role": row.role,
        "model_id": str(row.model_id),
        "display_name": snapshot.get("display_name"),
        "served_model_name": snapshot.get("served_model_name"),
        "provider": snapshot.get("provider"),
        "status": snapshot.get("status"),
        "supports_image_generation": bool(snapshot.get("supports_image_generation")),
        "ready": _ready(row.role, snapshot),
        "row_version": int(row.row_version),
        "updated_at": row.updated_at.isoformat() if getattr(row.updated_at, "isoformat", None) else row.updated_at,
    }


def capabilities_from_bindings(bindings: list[dict[str, Any]]) -> dict[str, Any]:
    by_role = {item["role"]: item for item in bindings}

    def summarize(role: str, flag: str | None) -> dict[str, bool]:
        item = by_role.get(role)
        bound = item is not None
        verified = bool(item.get(flag)) if item is not None and flag else bound and bool(item and item.get("ready"))
        ready = bool(item and item.get("ready"))
        return {"bound": bound, "verified": verified, "ready": ready}

    return {"image_generation": summarize("IMAGE_GENERATION", "supports_image_generation")}


class ModelBindingService:
    def __init__(self, *, uow_factory, catalog_client):
        self._uow_factory = uow_factory
        self._catalog = catalog_client

    async def list(self, *, domain_id: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.bindings is not None
            return [binding_view(row) for row in await uow.bindings.list(domain_id=domain_id)]

    async def model_references(self, *, model_id: UUID) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.bindings is not None
            return [
                {
                    "service": "media-studio-app",
                    "domain_id": str(row.domain_id),
                    "resource_type": "media_model_binding",
                    "resource_id": str(row.binding_id),
                    "usage": row.role.lower(),
                }
                for row in await uow.bindings.model_references(model_id=model_id)
            ]

    async def upsert(self, command: UpsertBindingCommand) -> dict[str, Any]:
        catalog = await load_catalog_model(self._catalog, command.model_id)
        snapshot = safe_model_snapshot(catalog)
        if not is_role_ready(command.role, snapshot):
            raise MediaStudioApplicationError(
                "MODEL_CAPABILITY_UNVERIFIED",
                "只能绑定已启用且对应能力已验收的模型",
                status_code=422,
            )
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            assert uow.bindings is not None
            row = await uow.bindings.get_by_role(
                domain_id=command.domain_id, role=command.role, lock=True,
            )
            if row is None:
                row = MediaStudioModelBindingEntity(
                    binding_id=uuid7(),
                    domain_id=command.domain_id,
                    role=command.role,
                    model_id=command.model_id,
                    model_snapshot_json=snapshot,
                    created_by=command.actor_id,
                    updated_by=command.actor_id,
                    created_at=now,
                    updated_at=now,
                )
                await uow.bindings.add(row)
            else:
                if command.expected_row_version is not None and int(row.row_version) != command.expected_row_version:
                    raise MediaStudioApplicationError(
                        "STATE_VERSION_CONFLICT", "模型绑定版本已变化",
                    )
                row.model_id = command.model_id
                row.model_snapshot_json = snapshot
                row.updated_by = command.actor_id
                row.updated_at = now
                row.row_version = int(row.row_version) + 1
            await uow.commit()
            return binding_view(row)
