"""X Search Run 的创建、查询与来源投影。"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.exc import IntegrityError

from assistant_app.application.catalog import is_role_ready, load_catalog_model, safe_model_snapshot
from assistant_app.application.errors import AssistantApplicationError
from assistant_app.application.runtime import (
    RESEARCH_KIND, TERMINAL_STATUSES, append_event, event_view, not_found, run_view, source_view,
)
from assistant_app.entities import AssistantRunEntity
from platform_core.identity import uuid7


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateResearchCommand(_Model):
    domain_id: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    input: str = Field(min_length=1, max_length=32000)
    from_date: date | None = None
    to_date: date | None = None
    allowed_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    excluded_x_handles: tuple[str, ...] = Field(default=(), max_length=20)
    enable_image_understanding: bool = False
    enable_video_understanding: bool = False
    search_context_size: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"

    @model_validator(mode="after")
    def validate_filters(self) -> "CreateResearchCommand":
        if self.allowed_x_handles and self.excluded_x_handles:
            raise ValueError("allowed_x_handles 与 excluded_x_handles 不能同时设置")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date 不能晚于 to_date")
        return self


class ResearchRunService:
    def __init__(self, *, uow_factory, catalog_client):
        self._uow_factory = uow_factory
        self._catalog = catalog_client

    async def create(self, command: CreateResearchCommand) -> tuple[dict[str, Any], bool]:
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
        request_json = command.model_dump(
            mode="json",
            exclude={"domain_id", "actor_id", "request_id", "idempotency_key"},
        )
        run = AssistantRunEntity(
            run_id=run_id,
            domain_id=command.domain_id,
            kind=RESEARCH_KIND,
            status="ACCEPTED",
            actor_id=command.actor_id,
            request_id=command.request_id,
            trace_id=command.trace_id,
            idempotency_key=command.idempotency_key,
            model_id=model_id,
            model_snapshot_json=snapshot,
            request_json=request_json,
            created_at=now,
            updated_at=now,
        )
        try:
            async with self._uow_factory() as uow:
                assert uow.runs is not None
                await uow.runs.add(run)
                await append_event(uow, run, stage="ACCEPTED", message="已接受 X Search 请求")
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
            if row is None or row.kind != RESEARCH_KIND:
                not_found()
            if actor_id is not None and row.actor_id != actor_id:
                not_found()
            return run_view(row)

    async def list(self, *, domain_id: int, actor_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            rows = await uow.runs.list(
                domain_id=domain_id, kind=RESEARCH_KIND, status=status,
                actor_id=actor_id, limit=limit,
            )
            return [run_view(row) for row in rows]

    async def list_events(self, *, domain_id: int, run_id: UUID, actor_id: str | None = None) -> list[dict[str, Any]]:
        await self.get(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
        async with self._uow_factory() as uow:
            assert uow.run_events is not None
            return [event_view(row) for row in await uow.run_events.list(domain_id=domain_id, run_id=run_id)]

    async def list_sources(self, *, domain_id: int, run_id: UUID, actor_id: str | None = None) -> list[dict[str, Any]]:
        await self.get(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
        async with self._uow_factory() as uow:
            assert uow.x_sources is not None
            return [source_view(row) for row in await uow.x_sources.list(domain_id=domain_id, run_id=run_id)]

    async def delete(self, *, domain_id: int, run_id: UUID, actor_id: str | None = None) -> None:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            assert uow.run_events is not None
            assert uow.x_sources is not None
            row = await uow.runs.get(domain_id=domain_id, run_id=run_id, lock=True)
            if row is None or row.kind != RESEARCH_KIND:
                not_found()
            if actor_id is not None and row.actor_id != actor_id:
                not_found()
            await uow.run_events.delete_by_run(domain_id=domain_id, run_id=run_id)
            await uow.x_sources.delete_by_run(domain_id=domain_id, run_id=run_id)
            await uow.runs.delete(row)
            await uow.commit()

    async def _existing(self, *, domain_id: int, actor_id: str, idempotency_key: str) -> dict[str, Any] | None:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            row = await uow.runs.get_by_idempotency(
                domain_id=domain_id, actor_id=actor_id, idempotency_key=idempotency_key,
            )
            if row is None or row.kind != RESEARCH_KIND:
                return None
            return run_view(row)

    async def _bound_snapshot(self, domain_id: int) -> tuple[dict[str, Any], UUID]:
        async with self._uow_factory() as uow:
            assert uow.bindings is not None
            binding = await uow.bindings.get_by_role(domain_id=domain_id, role="X_SEARCH")
            if binding is None:
                raise AssistantApplicationError(
                    "MODEL_BINDING_MISSING", "当前 Domain 未绑定 X Search 模型", status_code=422,
                )
            model_id = binding.model_id
        catalog = await load_catalog_model(self._catalog, model_id)
        snapshot = safe_model_snapshot(catalog)
        if not is_role_ready("X_SEARCH", snapshot):
            raise AssistantApplicationError(
                "MODEL_CAPABILITY_UNVERIFIED", "X Search 模型未启用或尚未通过能力验收", status_code=422,
            )
        return snapshot, model_id
