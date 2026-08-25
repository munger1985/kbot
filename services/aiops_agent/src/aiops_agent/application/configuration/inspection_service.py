"""Inspection Plan 与 Plan Target 配置用例。"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from urllib.parse import urlparse
from uuid import UUID

from loguru import logger
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from aiops_agent.application.configuration.common import (
    ConfigurationScope,
    IdempotencyGuard,
    SignedCursorCodec,
    add_configuration_event,
    canonical_json,
    sha256_json,
)
from aiops_agent.application.configuration.schedule import (
    InspectionTemplateRegistry,
    next_cron_run,
)
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    resource_not_found,
    row_version_changed,
    state_conflict,
    validation_failed,
)
from aiops_agent.config import AIOpsManagementConfig
from aiops_agent.entities import (
    InspectionPlanEntity,
    InspectionTargetEntity,
    DiagnosticSourceEntity,
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetSourceBindingEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.ports.secret_store import SecretStorePort
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    AgentBindingCreate,
    AgentBindingPatch,
    AgentBindingView,
    HealthCheckReceipt,
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    InspectionPlanSummary,
    InspectionTargetCreate,
    InspectionTargetPatch,
    InspectionTargetView,
    SourceBindingCreate,
    SourceBindingPatch,
    SourceBindingView,
    DiagnosticSourceCreate,
    DiagnosticSourceDetail,
    DiagnosticSourcePage,
    DiagnosticSourcePatch,
    DiagnosticSourceSummary,
    PolicyCreate,
    PolicyDetail,
    PolicyPage,
    PolicySummary,
    SecretRefStatus,
    TargetCreate,
    TargetDetail,
    TargetPage,
    TargetPatch,
    TargetSummary,
    WebhookKeyRotation,
)
from platform_core.identity import uuid7

from .projections import (
    _target_detail,
    _target_summary,
    _agent_binding_view,
    _diagnostic_source_detail,
    _diagnostic_source_summary,
    _source_binding_view,
    _policy_detail,
    _policy_summary,
    _inspection_detail,
    _inspection_summary,
    _inspection_target_view,
)


class InspectionConfigurationMixin:
    def _validate_plan_definition(
        self,
        *,
        cron_expression: str,
        timezone: str,
        template_id: str,
        template_version: str,
        resolver_version: str,
    ) -> datetime:
        self._template_registry.validate(
            template_id=template_id,
            template_version=template_version,
            schedule_resolver_version=resolver_version,
        )
        return next_cron_run(
            expression=cron_expression,
            timezone_name=timezone,
        )

    async def create_inspection_plan(
        self,
        *,
        scope: ConfigurationScope,
        request: InspectionPlanCreate,
        idempotency_key: str,
    ) -> InspectionPlanDetail:
        self._validate_plan_definition(
            cron_expression=request.cron_expression,
            timezone=request.timezone,
            template_id=request.template_id,
            template_version=request.template_version,
            resolver_version=request.schedule_resolver_version,
        )

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> InspectionPlanDetail:
            assert uow.inspections is not None
            entity = InspectionPlanEntity(
                inspection_plan_id=uuid7(),
                domain_id=scope.domain_id,
                display_name=request.display_name,
                schedule_type=request.schedule_type,
                cron_expression=request.cron_expression,
                timezone=request.timezone,
                template_id=request.template_id,
                template_version=request.template_version,
                timeout_seconds=request.timeout_seconds,
                overlap_policy=request.overlap_policy,
                misfire_policy=request.misfire_policy,
                schedule_resolver_version=request.schedule_resolver_version,
                status="PAUSED",
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.inspections.add_plan(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="INSPECTION_PLAN",
                aggregate_id=entity.inspection_plan_id,
                event_type="INSPECTION_PLAN_CREATED",
                row_version=1,
            )
            return _inspection_detail(entity, active_target_count=0)

        return await self._idempotent(
            scope=scope,
            operation="INSPECTION_PLAN_CREATE",
            parent_resource="inspection-plans",
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=InspectionPlanDetail,
            handler=handler,
        )

    async def get_inspection_plan(
        self, *, scope: ConfigurationScope, plan_id: UUID
    ) -> InspectionPlanDetail:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            entity = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            if entity is None:
                raise resource_not_found("Inspection Plan")
            targets = await uow.inspections.list_active_targets(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            return _inspection_detail(
                entity, active_target_count=len(targets)
            )

    async def list_inspection_plans(
        self,
        *,
        scope: ConfigurationScope,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> InspectionPlanPage:
        if status is not None and status not in {
            "ACTIVE",
            "PAUSED",
            "DISABLED",
        }:
            raise validation_failed("Inspection Plan status 过滤条件无效")
        filters = {"status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._cursor_codec.decode(
                token=cursor, scope=scope, filters=filters
            )
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            entities = await uow.inspections.page_plans(
                domain_id=scope.domain_id,
                statuses=(status,) if status else None,
                before_updated_at=before_at,
                before_id=before_id,
                limit=limit + 1,
            )
            page_entities = entities[:limit]
            next_cursor = None
            if len(entities) > limit and page_entities:
                last = page_entities[-1]
                next_cursor = self._cursor_codec.encode(
                    scope=scope,
                    updated_at=last.updated_at,
                    resource_id=last.inspection_plan_id,
                    filters=filters,
                )
            return InspectionPlanPage(
                items=tuple(
                    _inspection_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def patch_inspection_plan(
        self,
        *,
        scope: ConfigurationScope,
        plan_id: UUID,
        request: InspectionPlanPatch,
        expected_version: int,
    ) -> InspectionPlanDetail:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            entity = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Inspection Plan")
            self._check_version(entity.row_version, expected_version)
            fields = request.model_dump(exclude_unset=True, mode="json")
            fields.pop("schema_version", None)
            if not fields:
                raise validation_failed("PATCH 至少需要一个可修改字段")
            definitions = {
                "cron_expression": fields.get(
                    "cron_expression", entity.cron_expression
                ),
                "timezone": fields.get("timezone", entity.timezone),
                "template_id": fields.get("template_id", entity.template_id),
                "template_version": fields.get(
                    "template_version", entity.template_version
                ),
                "resolver_version": fields.get(
                    "schedule_resolver_version",
                    entity.schedule_resolver_version,
                ),
            }
            next_run_at = self._validate_plan_definition(**definitions)
            for name, value in fields.items():
                setattr(entity, name, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            targets = await uow.inspections.list_active_targets(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            if entity.status == "ACTIVE":
                if not targets:
                    raise validation_failed(
                        "Active 计划必须至少保留一个 Active Target"
                    )
                if len(targets) > self._max_inspection_targets:
                    raise validation_failed("计划 Target 数超过部署限制")
                entity.next_run_at = next_run_at
            else:
                entity.next_run_at = None
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="INSPECTION_PLAN",
                aggregate_id=plan_id,
                event_type="INSPECTION_PLAN_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _inspection_detail(
                entity, active_target_count=len(targets)
            )
            await uow.commit()
            return response

    async def command_inspection_plan(
        self,
        *,
        scope: ConfigurationScope,
        plan_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> InspectionPlanDetail:
        transitions = {
            "activate": ({"PAUSED"}, "ACTIVE"),
            "pause": ({"ACTIVE", "DISABLED"}, "PAUSED"),
            "disable": ({"ACTIVE", "PAUSED"}, "DISABLED"),
        }
        if command not in transitions:
            raise validation_failed("未知 Inspection Plan 命令")
        allowed, destination = transitions[command]

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> InspectionPlanDetail:
            assert uow.inspections is not None
            entity = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Inspection Plan")
            self._check_version(entity.row_version, expected_version)
            if entity.status not in allowed:
                raise state_conflict(
                    f"Inspection Plan 不能从 {entity.status} 执行 {command}"
                )
            targets = await uow.inspections.list_active_targets(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            if destination == "ACTIVE":
                if not targets:
                    raise validation_failed("激活计划前至少需要一个 Active Target")
                if len(targets) > self._max_inspection_targets:
                    raise validation_failed("计划 Target 数超过部署限制")
                entity.next_run_at = self._validate_plan_definition(
                    cron_expression=entity.cron_expression,
                    timezone=entity.timezone,
                    template_id=entity.template_id,
                    template_version=entity.template_version,
                    resolver_version=entity.schedule_resolver_version,
                )
            else:
                entity.next_run_at = None
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="INSPECTION_PLAN",
                aggregate_id=plan_id,
                event_type=f"INSPECTION_PLAN_{destination}",
                row_version=int(entity.row_version),
            )
            return _inspection_detail(
                entity, active_target_count=len(targets)
            )

        return await self._idempotent(
            scope=scope,
            operation=f"INSPECTION_PLAN_{command.upper()}",
            parent_resource=str(plan_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=InspectionPlanDetail,
            handler=handler,
        )

    async def add_inspection_target(
        self,
        *,
        scope: ConfigurationScope,
        plan_id: UUID,
        request: InspectionTargetCreate,
        expected_plan_version: int,
        idempotency_key: str,
    ) -> InspectionTargetView:
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> InspectionTargetView:
            assert uow.inspections is not None
            assert uow.targets is not None
            plan = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if plan is None:
                raise resource_not_found("Inspection Plan")
            self._check_version(plan.row_version, expected_plan_version)
            target = await uow.targets.get_scoped(
                target_id=request.target_id,
                domain_id=scope.domain_id,
            )
            if target is None:
                raise resource_not_found("Target")
            registration = self._template_registry.validate(
                template_id=plan.template_id,
                template_version=plan.template_version,
                schedule_resolver_version=plan.schedule_resolver_version,
            )
            self._template_registry.validate_overrides(
                registration=registration,
                overrides=request.template_overrides,
            )
            existing = await uow.inspections.list_targets(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            if any(item.target_id == request.target_id for item in existing):
                raise state_conflict("Target 已加入该 Inspection Plan")
            if len(existing) >= self._max_inspection_targets:
                raise validation_failed("计划 Target 数超过部署限制")
            entity = InspectionTargetEntity(
                inspection_target_id=uuid7(),
                inspection_plan_id=plan_id,
                target_id=request.target_id,
                template_overrides_json=request.template_overrides,
                status="ACTIVE",
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.inspections.add_target(entity)
            plan.updated_by = scope.actor_id
            plan.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="INSPECTION_PLAN",
                aggregate_id=plan_id,
                event_type="INSPECTION_TARGET_ADDED",
                row_version=int(plan.row_version),
                details={
                    "inspection_target_id": str(entity.inspection_target_id),
                    "target_id": str(entity.target_id),
                },
            )
            return _inspection_target_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="INSPECTION_TARGET_ADD",
            parent_resource=str(plan_id),
            idempotency_key=idempotency_key,
            payload={
                "request": request.model_dump(mode="json"),
                "plan_row_version": expected_plan_version,
            },
            response_type=InspectionTargetView,
            handler=handler,
        )

    async def list_inspection_targets(
        self, *, scope: ConfigurationScope, plan_id: UUID
    ) -> tuple[InspectionTargetView, ...]:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            plan = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            if plan is None:
                raise resource_not_found("Inspection Plan")
            entities = await uow.inspections.list_targets(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
            )
            return tuple(_inspection_target_view(item) for item in entities)

    async def patch_inspection_target(
        self,
        *,
        scope: ConfigurationScope,
        plan_id: UUID,
        plan_target_id: UUID,
        request: InspectionTargetPatch,
        expected_plan_version: int,
    ) -> InspectionTargetView:
        async with self._uow_factory() as uow:
            assert uow.inspections is not None
            plan = await uow.inspections.get_plan_scoped(
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if plan is None:
                raise resource_not_found("Inspection Plan")
            self._check_version(plan.row_version, expected_plan_version)
            entity = await uow.inspections.get_target_scoped(
                inspection_target_id=plan_target_id,
                inspection_plan_id=plan_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Inspection Target")
            fields = request.model_dump(exclude_unset=True, mode="json")
            fields.pop("schema_version", None)
            if not fields:
                raise validation_failed("PATCH 至少需要一个可修改字段")
            if "template_overrides" in fields:
                registration = self._template_registry.validate(
                    template_id=plan.template_id,
                    template_version=plan.template_version,
                    schedule_resolver_version=plan.schedule_resolver_version,
                )
                self._template_registry.validate_overrides(
                    registration=registration,
                    overrides=request.template_overrides,
                )
                fields["template_overrides_json"] = fields.pop(
                    "template_overrides"
                )
            for name, value in fields.items():
                setattr(entity, name, value)
            now = datetime.now(UTC)
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            plan.updated_by = scope.actor_id
            plan.updated_at = now
            if plan.status == "ACTIVE":
                active_targets = await uow.inspections.list_active_targets(
                    inspection_plan_id=plan_id,
                    domain_id=scope.domain_id,
                )
                if not active_targets:
                    raise validation_failed(
                        "Active 计划必须至少保留一个 Active Target"
                    )
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="INSPECTION_PLAN",
                aggregate_id=plan_id,
                event_type="INSPECTION_TARGET_UPDATED",
                row_version=int(plan.row_version),
                details={"inspection_target_id": str(plan_target_id)},
            )
            response = _inspection_target_view(entity)
            await uow.commit()
            return response
