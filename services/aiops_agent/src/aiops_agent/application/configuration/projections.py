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
    DatabaseCredentialStatus,
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
    """只投影 Secret 引用的提供方和不可逆指纹。"""
    if reference is None:
        return SecretRefStatus(configured=False)
    return SecretRefStatus(
        configured=True,
        provider=urlparse(reference).scheme,
        fingerprint=hashlib.sha256(reference.encode("utf-8")).hexdigest()[:16],
    )


def _managed_credential_status(credential_id: UUID | None) -> SecretRefStatus:
    """只暴露托管凭据是否配置及不可逆指纹。"""
    if credential_id is None:
        return SecretRefStatus(configured=False)
    return SecretRefStatus(
        configured=True,
        provider="managed",
        fingerprint=hashlib.sha256(credential_id.bytes).hexdigest()[:16],
    )


def _credential_status(credential_id, entity: TargetEntity) -> DatabaseCredentialStatus:
    return DatabaseCredentialStatus(
        configured=credential_id is not None,
        credential_id=credential_id,
        key_version=None,
        updated_at=entity.updated_at.astimezone(UTC) if credential_id else None,
    )

def _target_summary(entity: TargetEntity) -> TargetSummary:
    pending = bool(
        entity.connectivity_check_request_id
        and entity.connectivity_check_requested_at
        and (
            entity.last_connectivity_check_at is None
            or entity.last_connectivity_check_at
            < entity.connectivity_check_requested_at
        )
    )
    return TargetSummary(
        target_id=entity.target_id,
        display_name=entity.display_name,
        db_type=entity.db_type,
        environment=entity.environment,
        status=entity.status,
        connectivity_status=entity.connectivity_status,
        observed_status=entity.observed_status,
        readonly_connection_enabled=bool(entity.readonly_connection_enabled),
        controlled_change_enabled=bool(entity.controlled_change_enabled),
        connectivity_check_pending=pending,
        diagnostic_credential_configured=bool(entity.diagnostic_credential_id),
        execution_credential_configured=bool(entity.execution_credential_id),
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _target_detail(entity: TargetEntity) -> TargetDetail:
    return TargetDetail(
        **_target_summary(entity).model_dump(),
        version_code=entity.version_code,
        db_role=entity.db_role,
        endpoint=entity.endpoint_json,
        diagnostic_credential=_credential_status(entity.diagnostic_credential_id, entity),
        execution_credential=_credential_status(entity.execution_credential_id, entity),
        security_level=int(entity.security_level),
        capabilities=entity.capabilities_json or {},
        connectivity_version=int(entity.connectivity_version),
        last_observed_at=(
            entity.last_observed_at.astimezone(UTC)
            if entity.last_observed_at
            else None
        ),
        last_connectivity_check_at=(
            entity.last_connectivity_check_at.astimezone(UTC)
            if entity.last_connectivity_check_at
            else None
        ),
        last_connectivity_success_at=(
            entity.last_connectivity_success_at.astimezone(UTC)
            if entity.last_connectivity_success_at
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
        allow_mutation=bool(entity.allow_mutation),
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

def _diagnostic_source_summary(entity: DiagnosticSourceEntity) -> DiagnosticSourceSummary:
    pending = bool(
        entity.connectivity_check_request_id
        and entity.connectivity_check_requested_at
        and (
            entity.last_connectivity_check_at is None
            or entity.last_connectivity_check_at
            < entity.connectivity_check_requested_at
        )
    )
    return DiagnosticSourceSummary(
        source_id=entity.diagnostic_source_id,
        display_name=entity.display_name,
        source_type=entity.source_type,
        adapter_id=entity.adapter_id,
        adapter_version=entity.adapter_version,
        status=entity.status,
        connectivity_status=entity.connectivity_status,
        connectivity_check_pending=pending,
        row_version=int(entity.row_version),
        updated_at=entity.updated_at.astimezone(UTC),
    )

def _diagnostic_source_detail(entity: DiagnosticSourceEntity) -> DiagnosticSourceDetail:
    return DiagnosticSourceDetail(
        **_diagnostic_source_summary(entity).model_dump(),
        endpoint=entity.endpoint,
        secret=_managed_credential_status(entity.auth_credential_id),
        webhook_secret=_managed_credential_status(entity.webhook_credential_id),
        tls_profile=_secret_status(entity.tls_profile_ref),
        declared_capabilities=dict(entity.declared_capabilities_json or {}),
        discovered_capabilities=dict(entity.discovered_capabilities_json or {}),
        config=dict(entity.config_json or {}),
        webhook_configured=entity.webhook_key_hash is not None,
        connectivity_version=int(entity.connectivity_version),
        last_connectivity_check_at=(
            entity.last_connectivity_check_at.astimezone(UTC)
            if entity.last_connectivity_check_at
            else None
        ),
        last_connectivity_success_at=(
            entity.last_connectivity_success_at.astimezone(UTC)
            if entity.last_connectivity_success_at
            else None
        ),
        last_error_code=entity.last_error_code,
        created_at=entity.created_at.astimezone(UTC),
        created_by=entity.created_by,
        updated_by=entity.updated_by,
    )

def _source_binding_view(entity: TargetSourceBindingEntity) -> SourceBindingView:
    return SourceBindingView(
        binding_id=entity.target_source_binding_id,
        target_id=entity.target_id,
        source_id=entity.diagnostic_source_id,
        source_locator_key=entity.source_locator_key,
        source_locator=dict(entity.source_locator_json or {}),
        role=entity.role,
        priority=int(entity.priority),
        capability_scope=entity.capability_scope_json,
        mapping_overrides=entity.mapping_overrides_json,
        query_budget=entity.query_budget_json,
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
