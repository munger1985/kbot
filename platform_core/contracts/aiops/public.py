"""Main API 对外发布的 AIOps 请求与响应契约。"""

from __future__ import annotations

from typing import Literal

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
    TargetEndpoint,
    TargetPage,
    TargetPatch,
    TargetSummary,
    WebhookKeyRotation,
)
from .types import (
    AIOpsContract,
    ArtifactRef,
    HitlStatus,
    HitlType,
    JsonObject,
    OpsRunStatus,
    ProposalStatus,
    PUBLIC_SCHEMA_VERSION,
    ReportStatus,
    ReportType,
    ResultFormat,
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
    status: OpsRunStatus
    root_cause_grade: RootCauseGrade | None = None
    source_proposal_id: UUIDv7 | None = None
    source_result_artifact_id: UUIDv7 | None = None
    final_artifact: ArtifactRef | None = None
    row_version: int = Field(ge=1)
    created_at: UtcDatetime
    completed_at: UtcDatetime | None = None


class MonitoringEventReceipt(AIOpsContract):
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
