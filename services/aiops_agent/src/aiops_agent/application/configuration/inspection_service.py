"""由 DBA Agent 驱动的 Inspection Plan 配置用例。"""

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
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    InspectionPlanSummary,
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
)


class InspectionConfigurationMixin:
    async def _active_inspection_agent(self, *, uow, domain_id: int, agent_id: UUID):
        """确认计划引用的是已启用 DBA Agent，不解析其执行资源。"""
        assert uow.agents is not None
        binding = await uow.agents.get_active(
            domain_id=domain_id,
            agent_id=agent_id,
        )
        if binding is None:
            raise validation_failed("巡检计划必须选择一个已启用的 Agent")
        return binding

    async def _inspection_agent_target_count(
        self, *, uow, domain_id: int, agent_id: UUID
    ) -> int:
        """返回计划所选 Agent 当前版本关联的 Target 数。"""
        assert uow.agents is not None
        agent = await uow.agents.get(domain_id=domain_id, agent_id=agent_id)
        if agent is None or agent.current_version_id is None:
            return 0
        return len(
            await uow.agents.version_target_ids(
                agent_version_id=agent.current_version_id
            )
        )

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
        next_run_at = self._validate_plan_definition(
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
            binding = await self._active_inspection_agent(
                uow=uow,
                domain_id=scope.domain_id,
                agent_id=request.agent_id,
            )
            target_count = len(binding.target_ids)
            entity = InspectionPlanEntity(
                inspection_plan_id=uuid7(),
                domain_id=scope.domain_id,
                display_name=request.display_name,
                agent_id=request.agent_id,
                schedule_type=request.schedule_type,
                cron_expression=request.cron_expression,
                timezone=request.timezone,
                template_id=request.template_id,
                template_version=request.template_version,
                timeout_seconds=request.timeout_seconds,
                overlap_policy=request.overlap_policy,
                misfire_policy=request.misfire_policy,
                schedule_resolver_version=request.schedule_resolver_version,
                status="ACTIVE",
                next_run_at=next_run_at,
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
            return _inspection_detail(
                entity, agent_target_count=target_count
            )

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
            target_count = await self._inspection_agent_target_count(
                uow=uow,
                domain_id=scope.domain_id,
                agent_id=entity.agent_id,
            )
            return _inspection_detail(
                entity, agent_target_count=target_count
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
            fields = request.model_dump(exclude_unset=True, mode="python")
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
            binding = await self._active_inspection_agent(
                uow=uow,
                domain_id=scope.domain_id,
                agent_id=entity.agent_id,
            )
            target_count = len(binding.target_ids)
            if entity.status == "ACTIVE":
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
                entity, agent_target_count=target_count
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
            if destination == "ACTIVE":
                binding = await self._active_inspection_agent(
                    uow=uow,
                    domain_id=scope.domain_id,
                    agent_id=entity.agent_id,
                )
                target_count = len(binding.target_ids)
                entity.next_run_at = self._validate_plan_definition(
                    cron_expression=entity.cron_expression,
                    timezone=entity.timezone,
                    template_id=entity.template_id,
                    template_version=entity.template_version,
                    resolver_version=entity.schedule_resolver_version,
                )
            else:
                target_count = await self._inspection_agent_target_count(
                    uow=uow,
                    domain_id=scope.domain_id,
                    agent_id=entity.agent_id,
                )
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
                entity, agent_target_count=target_count
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
