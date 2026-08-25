"""Main API 对外发布的 AIOps 请求与响应契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .configuration import (
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
    NotificationSubscriptionList,
    NotificationSubscriptionUpsert,
    NotificationSubscriptionView,
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
    DatabaseCredentialInput,
    DatabaseCredentialStatus,
    TargetCreate,
    TargetConnectionTest,
    TargetConnectionTestResult,
    TargetDetail,
    TargetEndpoint,
    TargetPage,
    TargetPatch,
    TargetSummary,
    WebhookKeyRotation,
)
from .types import (
    AIOpsContract,
    ArtifactRef,
    CursorPage,
    HitlStatus,
    HitlType,
    JsonObject,
    OpsRunStatus,
    ProposalStatus,
    PUBLIC_SCHEMA_VERSION,
    ReportStatus,
    ReportType,
    ResultStatus,
    RootCauseGrade,
    Sha256Digest,
    TriggerType,
    UUIDv7,
    UtcDatetime,
)



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


class InspectionFirePage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[InspectionFireSummary, ...] = ()


class OpsRunCreate(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    agent_id: UUIDv7
    target_id: UUIDv7
    input: str = Field(min_length=1, max_length=32000)
    session_id: str | None = Field(default=None, max_length=256)
    observation_start: UtcDatetime | None = None
    observation_end: UtcDatetime | None = None
    client_metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_observation_window(self) -> "OpsRunCreate":
        if (self.observation_start is None) != (
            self.observation_end is None
        ):
            raise ValueError("观测窗口起止时间必须同时提供")
        if (
            self.observation_start is not None
            and self.observation_end is not None
            and self.observation_start >= self.observation_end
        ):
            raise ValueError("观测窗口结束时间必须晚于开始时间")
        return self


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
    interaction_mode: Literal["INTERACTIVE", "AUTONOMOUS"]
    investigation_mode: Literal[
        "INCIDENT", "ON_DEMAND", "INSPECTION", "VERIFICATION"
    ]
    status: OpsRunStatus
    root_cause_grade: RootCauseGrade | None = None
    source_proposal_id: UUIDv7 | None = None
    source_result_artifact_id: UUIDv7 | None = None
    final_artifact: ArtifactRef | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    completed_at: UtcDatetime | None = None


class OpsRunPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[OpsRunSummary, ...] = ()


class SignalEventSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    signal_event_id: UUIDv7
    diagnostic_source_id: UUIDv7
    source_event_key: str
    signal_kind: str
    event_class: str
    severity: str
    normalized_status: str
    summary: str | None = None
    occurred_at: UtcDatetime


class SituationSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    situation_id: UUIDv7
    target_id: UUIDv7
    situation_type: str
    title: str
    summary: str | None = None
    status: str
    severity: str
    event_count: int = Field(ge=0)
    row_version: int = Field(ge=1)
    first_observed_at: UtcDatetime
    last_observed_at: UtcDatetime
    resolved_at: UtcDatetime | None = None


class SituationView(SituationSummary):
    signal_events: tuple[SignalEventSummary, ...] = ()
    run_ids: tuple[UUIDv7, ...] = ()


class SituationPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[SituationSummary, ...] = ()


class OpsRunResult(AIOpsContract):
    """对外返回 Run 的最终 Artifact 内容。"""

    schema_version: str = PUBLIC_SCHEMA_VERSION
    ops_run_id: UUIDv7
    status: OpsRunStatus
    root_cause_grade: RootCauseGrade | None = None
    final_artifact: ArtifactRef | None = None
    payload: Any | None = None
    completed_at: UtcDatetime | None = None


class SignalEventReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    receipt_id: UUIDv7
    accepted: bool
    duplicate: bool = False
    event_count: int = Field(ge=0)


class PendingInputView(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    hitl_id: UUIDv7
    ops_run_id: UUIDv7
    hitl_type: HitlType
    status: HitlStatus
    request_artifact: ArtifactRef
    request: JsonObject | None = None
    expires_at: UtcDatetime
    row_version: int = Field(ge=1)


class HitlResponseItem(AIOpsContract):
    query_id: str = Field(min_length=1, max_length=256)
    status: ResultStatus
    raw_output: str | None = Field(default=None, max_length=65536)
    error: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_content(self) -> "HitlResponseItem":
        if self.status == ResultStatus.SUCCEEDED and not self.raw_output:
            raise ValueError("成功的 HITL 结果必须包含原始数据库输出")
        if self.status != ResultStatus.SUCCEEDED and self.raw_output:
            raise ValueError("失败或跳过的 HITL 结果不能包含数据库输出")
        return self


class HitlResponse(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    responses: tuple[HitlResponseItem, ...] = Field(min_length=1)
    note: str | None = Field(default=None, max_length=2000)


class HitlSkipCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)


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
    target_version: int = Field(ge=1)
    mode: str
    action_template_id: str
    action_template_version: str
    action_template_hash: Sha256Digest
    parameters: JsonObject
    parameter_fact_refs: dict[str, str]
    command_preview: str
    command_hash: Sha256Digest
    impact: str
    risk: str
    prerequisites: tuple[str, ...]
    rollback_plan: str
    verification_plan: str
    evidence_refs: tuple[str, ...] = ()
    proposal_hash: Sha256Digest
    status: ProposalStatus
    expires_at: UtcDatetime | None = None
    row_version: int = Field(ge=1)


class ProposalPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[ProposalView, ...] = ()


class ApprovalCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    expected_proposal_hash: Sha256Digest
    note: str | None = Field(default=None, max_length=2000)


class ApprovalReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    proposal_id: UUIDv7
    proposal_status: Literal["APPROVED"]
    approval_token_id: UUIDv7
    execution_id: UUIDv7
    execution_status: Literal["CREATED"]
    authorization_expires_at: UtcDatetime


class RejectionCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=2000)


class ManualResultCommand(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    expected_row_version: int = Field(ge=1)
    status: Literal["EXECUTED", "FAILED", "CANCELLED"]
    occurred_at: UtcDatetime
    note: str | None = Field(default=None, max_length=4000)
    bounded_output: str | None = Field(default=None, max_length=16000)


class ManualResultReceipt(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    proposal_id: UUIDv7
    status: str
    result_artifact: ArtifactRef


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


class ReportPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[ReportSummary, ...] = ()


class ReportVersionSummary(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    report_id: UUIDv7
    report_version: int = Field(ge=1)
    status: ReportStatus
    published_at: UtcDatetime | None = None


class ReportVersionPage(CursorPage):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    items: tuple[ReportVersionSummary, ...] = ()


class UploadSession(AIOpsContract):
    schema_version: str = PUBLIC_SCHEMA_VERSION
    upload_id: str = Field(min_length=1, max_length=256)
    upload_url: str
    required_headers: dict[str, str] = Field(default_factory=dict)
    max_bytes: int = Field(gt=0)
    expires_at: UtcDatetime
    expected_content_types: tuple[str, ...] = ()
    content_hash: Sha256Digest | None = None
