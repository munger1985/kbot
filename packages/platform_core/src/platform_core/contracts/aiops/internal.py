"""Main API、Agent Runtime、监控接入与 AIOps API 的内部契约。"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from .events import SafeAIOpsEvent
from .types import (
    AIOpsContract,
    ArtifactRef,
    CommandType,
    INTERNAL_SCHEMA_VERSION,
    JsonObject,
    OpsRunStatus,
    Sha256Digest,
    TriggerType,
    UUIDv7,
    UtcDatetime,
)


class CreateOpsRunCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    command_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=128)
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=256)
    agent_id: UUIDv7
    target_id: UUIDv7
    trigger_type: TriggerType
    input: str = Field(min_length=1, max_length=32000)
    session_id: str | None = Field(default=None, max_length=256)
    parent_agent_run_id: UUIDv7 | None = None
    parent_delegation_id: UUIDv7 | None = None
    trigger_event_id: UUIDv7 | None = None
    trigger_alert_id: UUIDv7 | None = None
    inspection_fire_id: UUIDv7 | None = None
    deadline: UtcDatetime | None = None
    blueprint_id: str = Field(
        default="kernel.observe-report", min_length=1, max_length=128
    )
    blueprint_version: str = Field(default="1", min_length=1, max_length=64)
    observation_start: UtcDatetime | None = None
    observation_end: UtcDatetime | None = None
    client_metadata: JsonObject = Field(default_factory=dict)


class OpsRunReceipt(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    ops_run_id: UUIDv7
    status: OpsRunStatus
    row_version: int = Field(ge=1)
    event_cursor: int = Field(ge=0)


class OpsRunQuery(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    ops_run_id: UUIDv7
    domain_id: str = Field(min_length=1, max_length=128)


class CancelRunCommand(AIOpsContract):
    command_type: Literal["CANCEL_RUN"] = "CANCEL_RUN"
    expected_row_version: int = Field(ge=1)
    reason: str | None = Field(default=None, max_length=1000)


class AnswerHitlCommand(AIOpsContract):
    command_type: Literal["ANSWER_HITL"] = "ANSWER_HITL"
    hitl_id: UUIDv7
    expected_row_version: int = Field(ge=1)
    response: JsonObject


class CancelHitlCommand(AIOpsContract):
    command_type: Literal["CANCEL_HITL"] = "CANCEL_HITL"
    hitl_id: UUIDv7
    expected_row_version: int = Field(ge=1)


class ApproveProposalCommand(AIOpsContract):
    command_type: Literal["APPROVE_PROPOSAL"] = "APPROVE_PROPOSAL"
    proposal_id: UUIDv7
    expected_row_version: int = Field(ge=1)
    expected_proposal_hash: Sha256Digest
    note: str | None = Field(default=None, max_length=2000)


class RejectProposalCommand(AIOpsContract):
    command_type: Literal["REJECT_PROPOSAL"] = "REJECT_PROPOSAL"
    proposal_id: UUIDv7
    expected_row_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=2000)


class RecordManualResultCommand(AIOpsContract):
    command_type: Literal["RECORD_MANUAL_RESULT"] = "RECORD_MANUAL_RESULT"
    proposal_id: UUIDv7
    expected_row_version: int = Field(ge=1)
    result: JsonObject


TypedOpsCommand = Annotated[
    Union[
        CancelRunCommand,
        AnswerHitlCommand,
        CancelHitlCommand,
        ApproveProposalCommand,
        RejectProposalCommand,
        RecordManualResultCommand,
    ],
    Field(discriminator="command_type"),
]


class OpsCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    command_id: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=256)
    ops_run_id: UUIDv7
    command: TypedOpsCommand


class MonitorWebhookEnvelope(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    request_id: str = Field(min_length=1, max_length=128)
    webhook_key_hash: Sha256Digest
    raw_body_base64: str | None = Field(default=None, max_length=30000000)
    raw_body_uri: str | None = Field(default=None, max_length=2048)
    raw_body_hash: Sha256Digest
    content_type: str = Field(min_length=1, max_length=256)
    signature_headers: dict[str, str] = Field(default_factory=dict)
    received_at: UtcDatetime

    @model_validator(mode="after")
    def validate_raw_body(self) -> "MonitorWebhookEnvelope":
        if bool(self.raw_body_base64) == bool(self.raw_body_uri):
            raise ValueError("Webhook 正文必须且只能使用 inline 或 URI")
        return self


class EventReceipt(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    event_id: UUIDv7
    accepted: bool
    duplicate: bool = False


class MonitorWebhookReceipt(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    inbox_id: UUIDv7
    accepted: bool
    duplicate: bool = False
    event_ids: tuple[UUIDv7, ...] = ()
    alert_ids: tuple[UUIDv7, ...] = ()


class FinalDiagnosisRef(AIOpsContract):
    artifact: ArtifactRef
    root_cause_grade: str
    report_id: UUIDv7 | None = None


class RootDelegationRequest(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    delegation_id: UUIDv7
    parent_agent_run_id: UUIDv7
    agent_id: UUIDv7
    target_id: UUIDv7
    domain_id: str = Field(min_length=1, max_length=128)
    user_intent: str = Field(min_length=1, max_length=32000)
    deadline: UtcDatetime


class RootDelegationReceipt(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    delegation_id: UUIDv7
    ops_run_id: UUIDv7
    status: OpsRunStatus
    child_event_cursor: int = Field(ge=0)


class DelegationEventPage(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    delegation_id: UUIDv7
    events: tuple[SafeAIOpsEvent, ...] = ()
    next_sequence: int = Field(ge=0)
    terminal: bool = False


class RootDelegationResult(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    delegation_id: UUIDv7
    ops_run_id: UUIDv7
    status: OpsRunStatus
    diagnosis: FinalDiagnosisRef | None = None
    safe_summary: str | None = Field(default=None, max_length=8000)
