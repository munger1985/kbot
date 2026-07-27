"""AIOps 配置 Entity 到安全 Wire DTO 的投影。"""

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

def _secret_status(reference: str | None) -> SecretRefStatus:
    if reference is None:
        return SecretRefStatus(configured=False)
    return SecretRefStatus(
        configured=True,
        provider=urlparse(reference).scheme,
        fingerprint=hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16],
    )

def _target_summary(entity: TargetEntity) -> TargetSummary:
    return TargetSummary(
        target_id=entity.target_id,
        target_key=entity.target_key,
        display_name=entity.display_name,
        db_type=entity.db_type,
        environment=entity.environment,
        execution_mode=entity.execution_mode,
        status=entity.status,
        health_status=entity.health_status,
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _target_detail(entity: TargetEntity) -> TargetDetail:
    return TargetDetail(
        **_target_summary(entity).model_dump(),
        version_code=entity.version_code,
        db_role=entity.db_role,
        endpoint=entity.endpoint_json or {},
        diagnostic_secret=_secret_status(entity.diagnostic_secret_ref),
        execution_secret=_secret_status(entity.execution_secret_ref),
        security_level=int(entity.security_level),
        capabilities=entity.capabilities_json or {},
        health_version=int(entity.health_version),
        last_health_check_at=(
            entity.last_health_check_at.astimezone(UTC)
            if entity.last_health_check_at
            else None
        ),
        last_error_code=entity.last_error_code,
        created_at=entity.created_at.astimezone(UTC),
        created_by=entity.created_by,
        updated_by=entity.updated_by,
    )

def _agent_binding_view(entity: TargetBindingEntity) -> AgentBindingView:
    return AgentBindingView(
        binding_id=entity.binding_id,
        target_id=entity.target_id,
        agent_id=entity.agent_id,
        access_mode=entity.access_mode,
        policy_id=entity.policy_id,
        allowed_actions=tuple(entity.allowed_actions_json or ()),
        change_window=entity.change_window_json,
        max_daily_executions=(
            int(entity.max_daily_executions)
            if entity.max_daily_executions is not None
            else None
        ),
        status=entity.status,
        row_version=int(entity.row_version),
        created_at=entity.created_at.astimezone(UTC),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _monitor_summary(entity: MonitorSourceEntity) -> MonitorSourceSummary:
    pending = bool(
        entity.health_check_request_id
        and entity.health_check_requested_at
        and (
            entity.last_health_check_at is None
            or entity.last_health_check_at < entity.health_check_requested_at
        )
    )
    return MonitorSourceSummary(
        source_id=entity.monitor_source_id,
        source_key=entity.source_key,
        display_name=entity.display_name,
        source_type=entity.source_type,
        status=entity.status,
        health_status=entity.health_status,
        health_check_pending=pending,
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _monitor_detail(entity: MonitorSourceEntity) -> MonitorSourceDetail:
    return MonitorSourceDetail(
        **_monitor_summary(entity).model_dump(),
        endpoint=entity.endpoint or "",
        secret=_secret_status(entity.secret_ref),
        webhook_secret=_secret_status(entity.webhook_secret_ref),
        tls_profile=_secret_status(entity.tls_profile_ref),
        capabilities=entity.capabilities_json or {},
        webhook_configured=entity.webhook_key_hash is not None,
        health_version=int(entity.health_version),
        last_health_check_at=(
            entity.last_health_check_at.astimezone(UTC)
            if entity.last_health_check_at
            else None
        ),
        last_error_code=entity.last_error_code,
        created_at=entity.created_at.astimezone(UTC),
        created_by=entity.created_by,
        updated_by=entity.updated_by,
    )

def _monitor_binding_view(entity: TargetMonitorEntity) -> MonitorBindingView:
    return MonitorBindingView(
        binding_id=entity.target_monitor_id,
        target_id=entity.target_id,
        source_id=entity.monitor_source_id,
        external_target_key=entity.external_target_key,
        role=entity.role,
        priority=int(entity.priority),
        metric_scope=entity.metric_scope_json,
        mapping_overrides=entity.mapping_overrides_json,
        status=entity.status,
        health_status=entity.health_status,
        row_version=int(entity.row_version),
        created_at=entity.created_at.astimezone(UTC),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _policy_summary(entity: PolicyEntity) -> PolicySummary:
    return PolicySummary(
        policy_id=entity.policy_id,
        policy_key=entity.policy_key,
        version_no=int(entity.version_no),
        display_name=entity.display_name,
        policy_hash=entity.policy_hash,
        status=entity.status,
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _policy_detail(entity: PolicyEntity) -> PolicyDetail:
    return PolicyDetail(
        **_policy_summary(entity).model_dump(),
        rules=entity.rules_json,
        effective_at=(
            entity.effective_at.astimezone(UTC)
            if entity.effective_at
            else None
        ),
        retired_at=(
            entity.retired_at.astimezone(UTC) if entity.retired_at else None
        ),
        created_at=entity.created_at.astimezone(UTC),
        created_by=entity.created_by,
        updated_by=entity.updated_by,
    )

def _inspection_summary(
    entity: InspectionPlanEntity,
) -> InspectionPlanSummary:
    return InspectionPlanSummary(
        plan_id=entity.inspection_plan_id,
        plan_key=entity.plan_key,
        display_name=entity.display_name,
        schedule_type=entity.schedule_type,
        timezone=entity.timezone,
        status=entity.status,
        next_run_at=(
            entity.next_run_at.astimezone(UTC) if entity.next_run_at else None
        ),
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _inspection_detail(
    entity: InspectionPlanEntity,
    *,
    active_target_count: int,
) -> InspectionPlanDetail:
    return InspectionPlanDetail(
        **_inspection_summary(entity).model_dump(),
        cron_expression=entity.cron_expression,
        template_id=entity.template_id,
        template_version=entity.template_version,
        timeout_seconds=int(entity.timeout_seconds),
        overlap_policy=entity.overlap_policy,
        misfire_policy=entity.misfire_policy,
        schedule_resolver_version=entity.schedule_resolver_version,
        active_target_count=active_target_count,
        created_at=entity.created_at.astimezone(UTC),
        created_by=entity.created_by,
        updated_by=entity.updated_by,
    )

def _inspection_target_view(
    entity: InspectionTargetEntity,
) -> InspectionTargetView:
    return InspectionTargetView(
        plan_target_id=entity.inspection_target_id,
        plan_id=entity.inspection_plan_id,
        target_id=entity.target_id,
        template_overrides=entity.template_overrides_json,
        status=entity.status,
        created_at=entity.created_at.astimezone(UTC),
        updated_at=entity.updated_at.astimezone(UTC),
    )
