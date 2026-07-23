"""Main API 对外发布的 AIOps 请求与响应契约。"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .types import (
    AIOpsContract,
    ArtifactRef,
    CursorPage,
    DatabaseType,
    ExecutionMode,
    HitlStatus,
    HitlType,
    JsonObject,
    OpsRunStatus,
    ProposalStatus,
    PUBLIC_SCHEMA_VERSION,
    ReportStatus,
    ReportType,
    ResourceStatus,
    ResultFormat,
    ResultStatus,
    RootCauseGrade,
    SecretRef,
    Sha256Digest,
    TriggerType,
    UUIDv7,
    UtcDatetime,
)


class TargetEndpoint(AIOpsContract):
    host: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65535)
    service: str | None = Field(default=None, max_length=256)
    database: str | None = Field(default=None, max_length=256)


class TargetCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    target_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    db_type: DatabaseType
    version_code: str = Field(min_length=1, max_length=64)
    environment: str = Field(min_length=1, max_length=32)
    db_role: str = Field(min_length=1, max_length=32)
    endpoint: TargetEndpoint
    diagnostic_secret_ref: SecretRef | None = None
    execution_secret_ref: SecretRef | None = None
    execution_mode: ExecutionMode = ExecutionMode.MONITOR_ONLY
    security_level: int = Field(default=1, ge=0, le=99)
    metadata: JsonObject = Field(default_factory=dict)


class TargetPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    version_code: str | None = Field(default=None, min_length=1, max_length=64)
    environment: str | None = Field(default=None, min_length=1, max_length=32)
    db_role: str | None = Field(default=None, min_length=1, max_length=32)
    endpoint: TargetEndpoint | None = None
    diagnostic_secret_ref: SecretRef | None = None
    execution_secret_ref: SecretRef | None = None
    execution_mode: ExecutionMode | None = None
    security_level: int | None = Field(default=None, ge=0, le=99)
    metadata: JsonObject | None = None


class TargetView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    target_id: UUIDv7
    target_key: str
    display_name: str
    db_type: DatabaseType
    version_code: str
    environment: str
    db_role: str
    endpoint: TargetEndpoint
    has_diagnostic_secret: bool
    has_execution_secret: bool
    execution_mode: ExecutionMode
    security_level: int
    status: ResourceStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class TargetPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[TargetView, ...] = ()


class AgentBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    agent_id: UUIDv7
    capabilities: tuple[str, ...] = ()


class AgentBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    status: ResourceStatus
    capabilities: tuple[str, ...] | None = None


class AgentBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    agent_id: UUIDv7
    capabilities: tuple[str, ...] = ()
    status: ResourceStatus
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class MonitorSourceCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    provider_type: str = Field(pattern=r"^(PROMETHEUS|ZABBIX|OEM)$")
    endpoint: str = Field(min_length=1, max_length=2048)
    secret_ref: SecretRef
    provider_config: JsonObject = Field(default_factory=dict)


class MonitorSourcePatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    endpoint: str | None = Field(default=None, min_length=1, max_length=2048)
    secret_ref: SecretRef | None = None
    provider_config: JsonObject | None = None
    status: ResourceStatus | None = None


class MonitorSourceView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    source_key: str
    display_name: str
    provider_type: str
    endpoint: str
    has_secret: bool
    status: ResourceStatus
    row_version: int = Field(ge=1)
    webhook_key_hint: str | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class MonitorBindingCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    provider_object_ref: str = Field(min_length=1, max_length=512)
    priority: int = Field(default=100, ge=0, le=10000)
    metric_scope: JsonObject = Field(default_factory=dict)


class MonitorBindingPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    provider_object_ref: str | None = Field(
        default=None, min_length=1, max_length=512
    )
    priority: int | None = Field(default=None, ge=0, le=10000)
    metric_scope: JsonObject | None = None
    status: ResourceStatus | None = None


class MonitorBindingView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    binding_id: UUIDv7
    target_id: UUIDv7
    source_id: UUIDv7
    provider_object_ref: str
    priority: int
    metric_scope: JsonObject
    status: ResourceStatus
    row_version: int = Field(ge=1)


class InspectionPlanCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    plan_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    schedule_type: str = Field(pattern=r"^(DAILY|WEEKLY|CRON)$")
    schedule_expression: str = Field(min_length=1, max_length=256)
    timezone: str = Field(min_length=1, max_length=64)
    target_ids: tuple[UUIDv7, ...] = Field(min_length=1)


class InspectionPlanPatch(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    schedule_expression: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    timezone: str | None = Field(default=None, min_length=1, max_length=64)
    target_ids: tuple[UUIDv7, ...] | None = None
    status: ResourceStatus | None = None


class InspectionPlanView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    plan_id: UUIDv7
    plan_key: str
    display_name: str
    schedule_type: str
    schedule_expression: str
    timezone: str
    target_ids: tuple[UUIDv7, ...]
    status: ResourceStatus
    next_run_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)


class InspectionFireSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    fire_id: UUIDv7
    plan_id: UUIDv7
    scheduled_at: UtcDatetime
    status: str
    target_count: int = Field(ge=0)
    completed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)


class InspectionFireView(InspectionFireSummary):
    run_ids: tuple[UUIDv7, ...] = ()
    created_at: UtcDatetime
    completed_at: UtcDatetime | None = None


class OpsRunCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    agent_id: UUIDv7
    target_id: UUIDv7
    input: str = Field(min_length=1, max_length=32000)
    session_id: str | None = Field(default=None, max_length=256)
    client_metadata: JsonObject = Field(default_factory=dict)


class OpsRunReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    ops_run_id: UUIDv7
    status: OpsRunStatus
    row_version: int = Field(ge=1)
    event_cursor: int = Field(ge=0)
    events_url: str


class OpsRunSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    ops_run_id: UUIDv7
    agent_id: UUIDv7
    target_id: UUIDv7
    trigger_type: TriggerType
    status: OpsRunStatus
    root_cause_grade: RootCauseGrade | None = None
    final_artifact: ArtifactRef | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    completed_at: UtcDatetime | None = None


class PendingInputView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    hitl_id: UUIDv7
    ops_run_id: UUIDv7
    hitl_type: HitlType
    status: HitlStatus
    request_artifact: ArtifactRef
    expires_at: UtcDatetime
    row_version: int = Field(ge=1)


class HitlResponseItem(AIOpsContract):
    query_id: str = Field(min_length=1, max_length=256)
    status: ResultStatus
    format: ResultFormat
    upload_id: str | None = Field(default=None, max_length=256)
    inline_data: str | None = Field(default=None, max_length=65536)
    error: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_content(self) -> "HitlResponseItem":
        if self.upload_id and self.inline_data:
            raise ValueError("HITL 结果不能同时包含 upload_id 和 inline_data")
        if self.status == ResultStatus.SUCCEEDED and not (
            self.upload_id or self.inline_data
        ):
            raise ValueError("成功的 HITL 结果必须包含上传引用或内联数据")
        return self


class HitlResponse(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    responses: tuple[HitlResponseItem, ...] = Field(min_length=1)
    note: str | None = Field(default=None, max_length=2000)


class HitlResult(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    hitl_id: UUIDv7
    status: HitlStatus
    row_version: int = Field(ge=1)
    accepted_artifact: ArtifactRef | None = None


class ProposalView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    proposal_id: UUIDv7
    ops_run_id: UUIDv7
    target_id: UUIDv7
    action_template_id: str
    action_template_version: str
    command_preview: str
    impact: str
    risk: str
    prerequisites: tuple[str, ...]
    rollback_plan: str
    verification_plan: str
    proposal_hash: Sha256Digest
    status: ProposalStatus
    row_version: int = Field(ge=1)


class ApprovalCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    expected_proposal_hash: Sha256Digest
    note: str | None = Field(default=None, max_length=2000)


class RejectionCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=2000)


class ManualResultCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    status: ResultStatus
    note: str | None = Field(default=None, max_length=4000)
    result_artifact: ArtifactRef | None = None


class ReportSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    report_id: UUIDv7
    report_key: str
    report_type: ReportType
    report_version: int = Field(ge=1)
    status: ReportStatus
    target_id: UUIDv7
    period_start: UtcDatetime
    period_end: UtcDatetime
    summary: str


class ReportView(ReportSummary):
    content_artifact: ArtifactRef
    corrected_from_report_id: UUIDv7 | None = None
    published_at: UtcDatetime | None = None


class ReportVersionSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    report_id: UUIDv7
    report_version: int = Field(ge=1)
    status: ReportStatus
    published_at: UtcDatetime | None = None


class UploadSession(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    upload_id: str = Field(min_length=1, max_length=256)
    upload_url: str
    required_headers: dict[str, str] = Field(default_factory=dict)
    max_bytes: int = Field(gt=0)
    expires_at: UtcDatetime
    expected_content_types: tuple[str, ...] = ()
    content_hash: Sha256Digest | None = None


class WebhookKeyRotation(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    source_id: UUIDv7
    webhook_key: str = Field(min_length=32, max_length=256)
    created_at: UtcDatetime
