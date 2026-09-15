"""文生图 Run 的创建与查询。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError

from assistant_app.application.catalog import is_role_ready, load_catalog_model, safe_model_snapshot
from assistant_app.application.errors import AssistantApplicationError
from assistant_app.application.runtime import IMAGE_KIND, append_event, event_view, not_found, run_view
from assistant_app.entities import AssistantPromptRevisionEntity, AssistantRunEntity
from platform_core.identity import uuid7


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateImageGenerationCommand(_Model):
    domain_id: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=32000)
    aspect_ratio: str = Field(default="1:1", pattern=r"^[1-9][0-9]?:[1-9][0-9]?$")
    count: int = Field(default=1, ge=1, le=4)


class ImageGenerationService:
    def __init__(self, *, uow_factory, catalog_client):
        self._uow_factory = uow_factory
        self._catalog = catalog_client

    async def create(self, command: CreateImageGenerationCommand) -> tuple[dict[str, Any], bool]:
        existing = await self._existing(
            domain_id=command.domain_id,
            actor_id=command.actor_id,
            idempotency_key=command.idempotency_key,
        )
        if existing is not None:
            return existing, False
        snapshot, model_id = await self._bound_snapshot(command.domain_id)
        now = datetime.now(timezone.utc)
        run_id = uuid7()
        revision_id = uuid7()
        request_json = command.model_dump(
            mode="json",
            exclude={"domain_id", "actor_id", "request_id", "idempotency_key"},
        )
        run = AssistantRunEntity(
            run_id=run_id,
            domain_id=command.domain_id,
            kind=IMAGE_KIND,
            status="ACCEPTED",
            actor_id=command.actor_id,
            request_id=command.request_id,
            trace_id=command.trace_id,
            idempotency_key=command.idempotency_key,
            model_id=model_id,
            model_snapshot_json=snapshot,
            request_json=request_json,
            prompt_revision_id=revision_id,
            created_at=now,
            updated_at=now,
        )
        revision = AssistantPromptRevisionEntity(
            prompt_revision_id=revision_id,
            domain_id=command.domain_id,
            run_id=run_id,
            actor_id=command.actor_id,
            prompt=command.prompt,
            aspect_ratio=command.aspect_ratio,
            image_count=command.count,
            version_no=1,
            created_at=now,
        )
        try:
            async with self._uow_factory() as uow:
                assert uow.runs is not None
                assert uow.prompt_revisions is not None
                await uow.runs.add(run)
                await uow.prompt_revisions.add(revision)
                await append_event(uow, run, stage="ACCEPTED", message="已接受文生图请求")
                await uow.commit()
        except IntegrityError:
            recovered = await self._existing(
                domain_id=command.domain_id,
                actor_id=command.actor_id,
                idempotency_key=command.idempotency_key,
            )
            if recovered is None:
                raise AssistantApplicationError(
                    "RUN_IDEMPOTENCY_CONFLICT", "无法恢复幂等 Run",
                )
            return recovered, False
        return run_view(run), True

    async def get(self, *, domain_id: int, run_id: UUID, actor_id: str | None = None) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            row = await uow.runs.get(domain_id=domain_id, run_id=run_id)
            if row is None or row.kind != IMAGE_KIND:
                not_found()
            if actor_id is not None and row.actor_id != actor_id:
                not_found()
            return run_view(row)

    async def list(self, *, domain_id: int, actor_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            rows = await uow.runs.list(
                domain_id=domain_id, kind=IMAGE_KIND, status=status,
                actor_id=actor_id, limit=limit,
            )
            return [run_view(row) for row in rows]

    async def list_events(self, *, domain_id: int, run_id: UUID, actor_id: str | None = None) -> list[dict[str, Any]]:
        await self.get(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
        async with self._uow_factory() as uow:
            assert uow.run_events is not None
            return [event_view(row) for row in await uow.run_events.list(domain_id=domain_id, run_id=run_id)]

    async def _existing(self, *, domain_id: int, actor_id: str, idempotency_key: str) -> dict[str, Any] | None:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            row = await uow.runs.get_by_idempotency(
                domain_id=domain_id, actor_id=actor_id, idempotency_key=idempotency_key,
            )
            if row is None or row.kind != IMAGE_KIND:
                return None
            return run_view(row)

    async def _bound_snapshot(self, domain_id: int) -> tuple[dict[str, Any], UUID]:
        async with self._uow_factory() as uow:
            assert uow.bindings is not None
            binding = await uow.bindings.get_by_role(domain_id=domain_id, role="IMAGE_GENERATION")
            if binding is None:
                raise AssistantApplicationError(
                    "MODEL_BINDING_MISSING", "当前 Domain 未绑定文生图模型", status_code=422,
                )
            model_id = binding.model_id
        catalog = await load_catalog_model(self._catalog, model_id)
        snapshot = safe_model_snapshot(catalog)
        if not is_role_ready("IMAGE_GENERATION", snapshot):
            raise AssistantApplicationError(
                "MODEL_CAPABILITY_UNVERIFIED", "文生图模型未启用或尚未通过能力验收", status_code=422,
            )
        return snapshot, model_id
