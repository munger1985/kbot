"""不可变 Policy 版本与生命周期配置用例。"""

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


class PolicyConfigurationMixin:
    @staticmethod
    def _validate_policy_rules(rules: dict[str, Any]) -> None:
        if rules.get("schema_version") != "ops.policy.v1":
            raise validation_failed("Policy rules.schema_version 必须为 ops.policy.v1")
        if not isinstance(rules.get("allow_agent_execution"), bool):
            raise validation_failed(
                "Policy rules.allow_agent_execution 必须为布尔值"
            )
        risk = rules.get("max_risk_level", "LOW")
        if risk not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise validation_failed("Policy max_risk_level 无效")
        actions = rules.get("allowed_action_types", [])
        if not isinstance(actions, list) or not all(
            isinstance(item, str) and item for item in actions
        ):
            raise validation_failed("Policy allowed_action_types 必须为字符串数组")
        minimum_severity = rules.get(
            "auto_observe_min_severity", "CRITICAL"
        )
        if minimum_severity not in {
            "INFO",
            "WARNING",
            "HIGH",
            "CRITICAL",
        }:
            raise validation_failed(
                "Policy auto_observe_min_severity 无效"
            )
        cooldown = rules.get("alert_cooldown_seconds", 900)
        if (
            not isinstance(cooldown, int)
            or isinstance(cooldown, bool)
            or not 0 <= cooldown <= 86400
        ):
            raise validation_failed(
                "Policy alert_cooldown_seconds 必须为 0 到 86400 的整数"
            )

    async def create_policy(
        self,
        *,
        scope: ConfigurationScope,
        request: PolicyCreate,
        idempotency_key: str,
    ) -> PolicyDetail:
        self._validate_policy_rules(request.rules)
        policy_hash = sha256_json(request.rules)

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> PolicyDetail:
            assert uow.policies is not None
            versions = await uow.policies.lock_versions(
                app_id=scope.app_id,
                domain_id=scope.domain_id,
                policy_key=request.policy_key,
            )
            entity = PolicyEntity(
                policy_id=uuid7(),
                app_id=scope.app_id,
                domain_id=scope.domain_id,
                policy_key=request.policy_key,
                version_no=max(
                    (int(item.version_no) for item in versions), default=0
                )
                + 1,
                display_name=request.display_name,
                rules_json=request.rules,
                policy_hash=policy_hash,
                status="DRAFT",
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            await uow.policies.add(entity)
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="POLICY",
                aggregate_id=entity.policy_id,
                event_type="POLICY_VERSION_CREATED",
                row_version=1,
                details={
                    "policy_key": request.policy_key,
                    "version_no": entity.version_no,
                    "policy_hash": policy_hash,
                },
            )
            return _policy_detail(entity)

        return await self._idempotent(
            scope=scope,
            operation="POLICY_CREATE_VERSION",
            parent_resource=request.policy_key,
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=PolicyDetail,
            handler=handler,
        )

    async def get_policy(
        self, *, scope: ConfigurationScope, policy_id: UUID
    ) -> PolicyDetail:
        async with self._uow_factory() as uow:
            assert uow.policies is not None
            entity = await uow.policies.get_scoped(
                policy_id=policy_id,
                app_id=scope.app_id,
                domain_id=scope.domain_id,
            )
            if entity is None:
                raise resource_not_found("Policy")
            return _policy_detail(entity)

    async def list_policies(
        self,
        *,
        scope: ConfigurationScope,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> PolicyPage:
        if status is not None and status not in {
            "DRAFT",
            "ACTIVE",
            "RETIRED",
        }:
            raise validation_failed("Policy status 过滤条件无效")
        filters = {"status": status}
        before_at = before_id = None
        if cursor:
            before_at, before_id = self._cursor_codec.decode(
                token=cursor, scope=scope, filters=filters
            )
        async with self._uow_factory() as uow:
            assert uow.policies is not None
            entities = await uow.policies.page_scoped(
                app_id=scope.app_id,
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
                    resource_id=last.policy_id,
                    filters=filters,
                )
            return PolicyPage(
                items=tuple(
                    _policy_summary(item) for item in page_entities
                ),
                next_cursor=next_cursor,
                has_more=len(entities) > limit,
            )

    async def command_policy(
        self,
        *,
        scope: ConfigurationScope,
        policy_id: UUID,
        command: str,
        expected_version: int,
        idempotency_key: str,
    ) -> PolicyDetail:
        if command not in {"activate", "retire"}:
            raise validation_failed("未知 Policy 命令")

        async def handler(
            uow: AIOpsUnitOfWork, now: datetime
        ) -> PolicyDetail:
            assert uow.policies is not None
            candidate = await uow.policies.get_scoped(
                policy_id=policy_id,
                app_id=scope.app_id,
                domain_id=scope.domain_id,
                lock=True,
            )
            if candidate is None:
                raise resource_not_found("Policy")
            self._check_version(candidate.row_version, expected_version)
            self._validate_policy_rules(candidate.rules_json)
            if sha256_json(candidate.rules_json) != candidate.policy_hash:
                raise state_conflict("Policy Hash 校验失败")
            versions = await uow.policies.lock_versions(
                app_id=scope.app_id,
                domain_id=scope.domain_id,
                policy_key=candidate.policy_key,
            )
            if command == "activate":
                if candidate.status != "DRAFT":
                    raise state_conflict("只有 DRAFT Policy 可以激活")
                for current in versions:
                    if current.status == "ACTIVE":
                        current.status = "RETIRED"
                        current.retired_at = now
                        current.updated_by = scope.actor_id
                        current.updated_at = now
                candidate.status = "ACTIVE"
                candidate.effective_at = now
                event = "POLICY_ACTIVATED"
            else:
                if candidate.status != "ACTIVE":
                    raise state_conflict("只有 ACTIVE Policy 可以退役")
                candidate.status = "RETIRED"
                candidate.retired_at = now
                event = "POLICY_RETIRED"
            candidate.updated_by = scope.actor_id
            candidate.updated_at = now
            await uow.session.flush()  # type: ignore[union-attr]
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="POLICY",
                aggregate_id=policy_id,
                event_type=event,
                row_version=int(candidate.row_version),
            )
            return _policy_detail(candidate)

        return await self._idempotent(
            scope=scope,
            operation=f"POLICY_{command.upper()}",
            parent_resource=str(policy_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version},
            response_type=PolicyDetail,
            handler=handler,
        )
