"""Target 与 Agent Binding 配置用例。"""

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
    MonitorSourceEntity,
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetMonitorEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from aiops_agent.ports.agent_runtime import AgentRuntimePort
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
    MonitorBindingCreate,
    MonitorBindingPatch,
    MonitorBindingView,
    MonitorSourceCreate,
    MonitorSourceDetail,
    MonitorSourcePage,
    MonitorSourcePatch,
    MonitorSourceSummary,
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
    _monitor_detail,
    _monitor_summary,
    _monitor_binding_view,
    _policy_detail,
    _policy_summary,
    _inspection_detail,
    _inspection_summary,
    _inspection_target_view,
)


class TargetConfigurationMixin:
    async def create_target(
        self,
        *,
        scope: ConfigurationScope,
        request: TargetCreate,
        idempotency_key: str,
    ) -> TargetDetail:
        await self._validate_secret_refs(
            request.diagnostic_secret_ref,
            request.execution_secret_ref,
        )
        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> TargetDetail:
            assert uow.targets is not None
            entity = TargetEntity(
                target_id=uuid7(),
                domain_id=scope.domain_id,
                display_name=request.display_name,
                db_type=request.db_type,
                version_code=request.version_code,
                environment=request.environment,
                db_role=request.db_role,
                endpoint_json=(
                    request.endpoint.model_dump(mode="json")
                    if request.endpoint is not None
                    else None
                ),
                diagnostic_secret_ref=request.diagnostic_secret_ref,
                execution_secret_ref=request.execution_secret_ref,
                security_level=request.security_level,
                capabilities_json=request.capabilities,
                status="MAINTENANCE",
                health_status="UNKNOWN",
                row_version=1,
                health_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.targets.add_target(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET",
                aggregate_id=entity.target_id,
                event_type="TARGET_CREATED",
                row_version=1,
            )
            return _target_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation="TARGET_CREATE",
            parent_resource="targets",
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=TargetDetail,
            handler=handler,
        )

    async def get_target(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> TargetDetail:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            entity = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            if entity is None:
                raise resource_not_found("Target")
            return _target_detail(entity)

    async def list_targets(
        self,
        *,
        scope: ConfigurationScope,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> TargetPage:
        if status is not None and status not in {
            "ACTIVE",
            "MAINTENANCE",
            "DISABLED",
        }:
            raise validation_failed("Target status 过滤条件无效")
        filters = {"status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._cursor_codec.decode(
                token=cursor,
                scope=scope,
                filters=filters,
            )
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            entities = await uow.targets.page_scoped(
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
                    resource_id=last.target_id,
                    filters=filters,
                )
            return TargetPage(
                items=tuple(
                    _target_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def patch_target(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        request: TargetPatch,
        expected_version: int,
    ) -> TargetDetail:
        fields = request.model_dump(exclude_unset=True, mode="json")
        fields.pop("schema_version", None)
        if not fields:
            raise validation_failed("PATCH 至少需要一个可修改字段")
        await self._validate_secret_refs(
            fields.get("diagnostic_secret_ref"),
            fields.get("execution_secret_ref"),
        )
        connectivity_changed = bool(
            {"endpoint", "diagnostic_secret_ref"} & fields.keys()
        )
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            entity = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Target")
            self._check_version(entity.row_version, expected_version)
            if "endpoint" in fields:
                endpoint = request.endpoint
                if endpoint is None:
                    raise validation_failed("Endpoint 不能为空")
                if entity.db_type == "ORACLE" and (
                    not endpoint.service or endpoint.database
                ):
                    raise validation_failed("Oracle Endpoint 必须只设置 service")
                if entity.db_type == "MYSQL" and (
                    not endpoint.database or endpoint.service
                ):
                    raise validation_failed("MySQL Endpoint 必须只设置 database")
                fields["endpoint_json"] = fields.pop("endpoint")
            for name, value in fields.items():
                setattr(entity, name, value)
            if connectivity_changed:
                if entity.status == "ACTIVE":
                    entity.status = "MAINTENANCE"
                entity.health_status = "UNKNOWN"
                entity.health_version = int(entity.health_version) + 1
                entity.last_health_check_at = None
                entity.last_error_code = None
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET",
                aggregate_id=target_id,
                event_type="TARGET_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _target_detail(entity)
            await uow.commit()
            return response

    async def command_target(
        self,
        *,
        scope: ConfigurationScope,
        target_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> TargetDetail:
        transitions = {
            "activate": ({"MAINTENANCE"}, "ACTIVE"),
            "maintenance": ({"ACTIVE", "DISABLED"}, "MAINTENANCE"),
            "disable": ({"ACTIVE", "MAINTENANCE"}, "DISABLED"),
        }
        if command not in transitions:
            raise validation_failed("未知 Target 状态命令")
        allowed, destination = transitions[command]

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> TargetDetail:
            assert uow.targets is not None
            entity = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Target")
            self._check_version(entity.row_version, expected_version)
            if entity.status not in allowed:
                raise state_conflict(
                    f"Target 不能从 {entity.status} 执行 {command}"
                )
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET",
                aggregate_id=target_id,
                event_type=f"TARGET_{destination}",
                row_version=int(entity.row_version),
            )
            return _target_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"TARGET_{command.upper()}",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload={"target_id": str(target_id), "row_version": expected_version},
            response_type=TargetDetail,
            handler=handler,
        )

    async def create_agent_binding(
        self,
        *,
        scope: ConfigurationScope,
        auth_context: AuthContext,
        target_id: UUID,
        request: AgentBindingCreate,
        idempotency_key: str,
    ) -> AgentBindingView:
        await self._agent_runtime.validate_aiops_agent(
            agent_id=request.agent_id,
            domain_id=scope.domain_id,
            auth_context=auth_context,
        )

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> AgentBindingView:
            assert uow.targets is not None
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if target is None:
                raise resource_not_found("Target")
            existing = await uow.targets.get_agent_binding(
                target_id=target_id,
                agent_id=request.agent_id,
                domain_id=scope.domain_id,
            )
            if existing:
                raise state_conflict("该 Agent 已绑定此 Target")
            await self._validate_policy_reference(
                uow=uow,
                scope=scope,
                policy_id=request.policy_id,
                require_active=False,
            )
            entity = TargetBindingEntity(
                binding_id=uuid7(),
                target_id=target_id,
                agent_id=request.agent_id,
                allow_mutation=request.allow_mutation,
                policy_id=request.policy_id,
                allowed_actions_json=list(request.allowed_actions),
                change_window_json=request.change_window,
                max_daily_executions=request.max_daily_executions,
                status="ACTIVE",
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.targets.add_binding(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_BINDING",
                aggregate_id=entity.binding_id,
                event_type="TARGET_BINDING_CREATED",
                row_version=1,
                details={"target_id": str(target_id)},
            )
            return _agent_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation="TARGET_BINDING_CREATE",
            parent_resource=str(target_id),
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=AgentBindingView,
            handler=handler,
        )

    async def _validate_policy_reference(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        policy_id: UUID | None,
        require_active: bool,
    ) -> None:
        if policy_id is None:
            if require_active:
                raise validation_failed("EXECUTE Binding 必须引用 Active Policy")
            return
        assert uow.policies is not None
        policy = await uow.policies.get_scoped(
            policy_id=policy_id,
            domain_id=scope.domain_id,
        )
        if policy is None:
            raise resource_not_found("Policy")
        if require_active and (
            policy.status != "ACTIVE"
            or policy.rules_json.get("allow_agent_execution") is not True
        ):
            raise validation_failed(
                "EXECUTE Binding 必须引用允许 Agent 执行的 Active Policy"
            )

    async def list_agent_bindings(
        self, *, scope: ConfigurationScope, target_id: UUID
    ) -> tuple[AgentBindingView, ...]:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            if target is None:
                raise resource_not_found("Target")
            entities = await uow.targets.list_agent_bindings(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            return tuple(_agent_binding_view(item) for item in entities)

    async def patch_agent_binding(
        self,
        *,
        scope: ConfigurationScope,
        auth_context: AuthContext,
        target_id: UUID,
        binding_id: UUID,
        request: AgentBindingPatch,
        expected_version: int,
    ) -> AgentBindingView:
        async with self._uow_factory() as uow:
            assert uow.targets is not None
            entity = await uow.targets.get_binding_scoped(
                binding_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Agent Binding")
            self._check_version(entity.row_version, expected_version)
            fields = request.model_dump(exclude_unset=True, mode="json")
            fields.pop("schema_version", None)
            if not fields:
                raise validation_failed("PATCH 至少需要一个可修改字段")
            policy_id = (
                request.policy_id
                if "policy_id" in request.model_fields_set
                else entity.policy_id
            )
            await self._validate_policy_reference(
                uow=uow,
                scope=scope,
                policy_id=policy_id,
                require_active=False,
            )
            if "allowed_actions" in fields:
                fields["allowed_actions_json"] = fields.pop("allowed_actions")
            if "change_window" in fields:
                fields["change_window_json"] = fields.pop("change_window")
            for name, value in fields.items():
                setattr(entity, name, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now(UTC)
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_BINDING",
                aggregate_id=binding_id,
                event_type="TARGET_BINDING_UPDATED",
                row_version=int(entity.row_version),
            )
            response = _agent_binding_view(entity)
            await uow.commit()
            return response

    async def command_agent_binding(
        self,
        *,
        scope: ConfigurationScope,
        auth_context: AuthContext,
        target_id: UUID,
        binding_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> AgentBindingView:
        destination = {"revoke": "REVOKED", "restore": "ACTIVE"}.get(command)
        if destination is None:
            raise validation_failed("未知 Agent Binding 命令")
        if command == "restore":
            async with self._uow_factory() as read_uow:
                assert read_uow.targets is not None
                entity = await read_uow.targets.get_binding_scoped(
                    binding_id=binding_id,
                    target_id=target_id,
                    domain_id=scope.domain_id,
                )
                if entity is None:
                    raise resource_not_found("Agent Binding")
                agent_id = entity.agent_id
            await self._agent_runtime.validate_aiops_agent(
                agent_id=agent_id,
                domain_id=scope.domain_id,
                auth_context=auth_context,
            )

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> AgentBindingView:
            assert uow.targets is not None
            entity = await uow.targets.get_binding_scoped(
                binding_id=binding_id,
                target_id=target_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Agent Binding")
            self._check_version(entity.row_version, expected_version)
            expected_status = "REVOKED" if command == "restore" else "ACTIVE"
            if entity.status != expected_status:
                raise state_conflict(
                    f"Agent Binding 不能从 {entity.status} 执行 {command}"
                )
            if destination == "ACTIVE":
                await self._validate_policy_reference(
                    uow=uow,
                    scope=scope,
                    policy_id=entity.policy_id,
                    require_active=False,
                )
            entity.status = destination
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="TARGET_BINDING",
                aggregate_id=binding_id,
                event_type=f"TARGET_BINDING_{destination}",
                row_version=int(entity.row_version),
            )
            return _agent_binding_view(entity)

        return await self._idempotent(
            scope=scope,
            operation=f"TARGET_BINDING_{command.upper()}",
            parent_resource=str(binding_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=AgentBindingView,
            handler=handler,
        )
