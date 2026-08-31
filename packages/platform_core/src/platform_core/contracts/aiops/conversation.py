"""AIOps 专业 DBA 对话、Turn、证据和回答块契约。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .types import (
    AIOpsContract,
    CursorPage,
    JsonObject,
    MeasurementSemantics,
    SufficiencyStatus,
    UUIDv7,
    UtcDatetime,
)


CONVERSATION_SCHEMA_VERSION = "aiops.conversation.v1"
TURN_EVENT_SCHEMA_VERSION = "aiops.turn-event.v1"


class ConversationStatus(StrEnum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    ARCHIVED = "ARCHIVED"


class ConversationSourceType(StrEnum):
    CHAT = "CHAT"
    SITUATION = "SITUATION"
    RUN = "RUN"
    REPORT = "REPORT"


class TurnStatus(StrEnum):
    QUEUED = "QUEUED"
    ACCEPTED = "ACCEPTED"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    COLLECTING = "COLLECTING"
    ASSESSING = "ASSESSING"
    REPLANNING = "REPLANNING"
    ANSWERING = "ANSWERING"
    WAITING_USER = "WAITING_USER"
    PROPOSAL_PENDING = "PROPOSAL_PENDING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DbaIntent(StrEnum):
    OBSERVE = "OBSERVE"
    DIAGNOSE = "DIAGNOSE"
    EXPLAIN = "EXPLAIN"
    PLAN = "PLAN"
    CHANGE = "CHANGE"
    VERIFY = "VERIFY"
    INSPECT = "INSPECT"


class AnswerBlockType(StrEnum):
    MARKDOWN = "MARKDOWN"
    TABLE = "TABLE"
    CHART = "CHART"
    EVIDENCE_REFERENCES = "EVIDENCE_REFERENCES"
    CLARIFICATION = "CLARIFICATION"
    EVIDENCE_REQUEST = "EVIDENCE_REQUEST"
    PROPOSAL_SUMMARY = "PROPOSAL_SUMMARY"
    VERIFICATION_COMPARISON = "VERIFICATION_COMPARISON"


class EvidenceRole(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    CONTEXT = "CONTEXT"
    USER_PROVIDED = "USER_PROVIDED"


class FreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class ConversationSourceContext(AIOpsContract):
    source_type: ConversationSourceType = ConversationSourceType.CHAT
    situation_id: UUIDv7 | None = None
    run_id: UUIDv7 | None = None
    report_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ConversationSourceContext":
        expected = {
            ConversationSourceType.CHAT: None,
            ConversationSourceType.SITUATION: self.situation_id,
            ConversationSourceType.RUN: self.run_id,
            ConversationSourceType.REPORT: self.report_id,
        }[ConversationSourceType(self.source_type)]
        supplied = tuple(
            value
            for value in (self.situation_id, self.run_id, self.report_id)
            if value is not None
        )
        if self.source_type == ConversationSourceType.CHAT and supplied:
            raise ValueError("CHAT 会话不能携带来源资源")
        if self.source_type != ConversationSourceType.CHAT and (
            expected is None or len(supplied) != 1
        ):
            raise ValueError("来源类型必须且只能携带对应的来源资源")
        return self


class ConversationCreate(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    agent_id: UUIDv7
    target_id: UUIDv7
    title: str | None = Field(default=None, min_length=1, max_length=256)
    source: ConversationSourceContext = Field(
        default_factory=ConversationSourceContext
    )


class ConversationStart(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    conversation: ConversationCreate
    first_turn: "TurnCreate"


class ConversationReceipt(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    conversation_id: UUIDv7
    agent_id: UUIDv7
    agent_version_id: UUIDv7
    target_id: UUIDv7
    status: ConversationStatus
    title: str | None = None
    created_at: UtcDatetime


class ConversationSummary(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    conversation_id: UUIDv7
    agent_id: UUIDv7
    agent_version_id: UUIDv7
    target_id: UUIDv7
    title: str | None = Field(default=None, max_length=256)
    status: ConversationStatus
    source_type: ConversationSourceType
    source_situation_id: UUIDv7 | None = None
    source_run_id: UUIDv7 | None = None
    source_report_id: UUIDv7 | None = None
    last_turn_no: int = Field(ge=0)
    created_at: UtcDatetime
    updated_at: UtcDatetime


class TurnCreate(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    content: tuple["InputContent", ...] = Field(min_length=1, max_length=16)
    idempotency_key: str = Field(min_length=1, max_length=128)
    source_run_id: UUIDv7 | None = None


class TurnReceipt(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    conversation_id: UUIDv7
    turn_id: UUIDv7
    turn_no: int = Field(ge=1)
    status: TurnStatus
    event_cursor: int = Field(ge=0)
    created_at: UtcDatetime


class AnswerCitationView(AIOpsContract):
    citation_no: int = Field(ge=1)
    turn_evidence_id: UUIDv7
    label: str = Field(min_length=1, max_length=256)


class AnswerBlockView(AIOpsContract):
    answer_block_id: UUIDv7
    block_no: int = Field(ge=1)
    block_type: AnswerBlockType
    schema_version: str = Field(min_length=1, max_length=64)
    payload: JsonObject
    content_hash: str = Field(min_length=64, max_length=64)
    citations: tuple[AnswerCitationView, ...] = ()


class ConversationMessageView(AIOpsContract):
    message_id: UUIDv7
    sequence_no: int = Field(ge=1)
    role: str = Field(min_length=1, max_length=16)
    message_type: str = Field(min_length=1, max_length=32)
    payload_schema: str = Field(min_length=1, max_length=64)
    payload: JsonObject
    artifact_id: UUIDv7 | None = None
    created_at: UtcDatetime


class TurnEvidenceGapView(AIOpsContract):
    """对话页可安全展示的未取证摘要，不包含原始证据行。"""

    source_id: str | None = Field(default=None, max_length=128)
    step_id: str | None = Field(default=None, max_length=128)
    code: str | None = Field(default=None, max_length=128)
    detail: str | None = Field(default=None, max_length=2000)
    retryable: bool = False


class TurnInvestigationActionView(AIOpsContract):
    """不暴露SQL和参数的用户可见调查动作。"""

    ordinal: int = Field(ge=1)
    action_id: str = Field(pattern=r"^a[0-9]+$")
    question: str = Field(min_length=1, max_length=2000)
    tool_id: str = Field(min_length=1, max_length=128)
    tool_class: str = Field(min_length=1, max_length=32)
    measurement_semantics: MeasurementSemantics
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    execution_mode: str = Field(pattern=r"^(AUTO_EXECUTE|APPROVAL_REQUIRED)$")
    status: str = Field(min_length=1, max_length=24)


class TurnInvestigationPlanView(AIOpsContract):
    """当前Turn已经验证并冻结的安全计划摘要。"""

    revision_no: int = Field(ge=1)
    actions: tuple[TurnInvestigationActionView, ...] = ()


class TurnSummary(AIOpsContract):
    turn_id: UUIDv7
    conversation_id: UUIDv7
    turn_no: int = Field(ge=1)
    status: TurnStatus
    resolved_target_id: UUIDv7 | None = None
    current_plan_revision: int = Field(default=0, ge=0)
    investigation_round: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    sufficiency_status: SufficiencyStatus | None = None
    evidence_gaps: tuple[TurnEvidenceGapView, ...] = ()
    event_cursor: int = Field(ge=0)
    error_domain: str | None = Field(default=None, max_length=32)
    error_code: str | None = Field(default=None, max_length=128)
    error_message: str | None = Field(default=None, max_length=2000)
    created_at: UtcDatetime
    completed_at: UtcDatetime | None = None


class TurnView(TurnSummary):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    messages: tuple[ConversationMessageView, ...] = ()
    answer_blocks: tuple[AnswerBlockView, ...] = ()
    investigation_plan: TurnInvestigationPlanView | None = None


class TurnPage(CursorPage):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    items: tuple[TurnSummary, ...] = ()


class TurnCancelCommand(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    reason: str | None = Field(default=None, max_length=1000)


class EvidenceResponseCreate(AIOpsContract):
    schema_version: str = CONVERSATION_SCHEMA_VERSION
    idempotency_key: str = Field(min_length=1, max_length=128)
    text: str | None = Field(default=None, max_length=32000)
    upload_id: UUIDv7 | None = None

    @model_validator(mode="after")
    def validate_response(self) -> "EvidenceResponseCreate":
        if (self.text is None) == (self.upload_id is None):
            raise ValueError("补证响应必须且只能提供文字或一个上传文件")
        return self


class TurnEventView(AIOpsContract):
    schema_version: str = TURN_EVENT_SCHEMA_VERSION
    turn_id: UUIDv7
    sequence_no: int = Field(ge=1)
    event_type: str = Field(min_length=1, max_length=64)
    payload: JsonObject
    occurred_at: UtcDatetime


class TurnEventPage(AIOpsContract):
    schema_version: str = TURN_EVENT_SCHEMA_VERSION
    events: tuple[TurnEventView, ...] = ()
    next_sequence: int = Field(ge=0)
    terminal: bool = False


from .investigation import InputContent  # noqa: E402

TurnCreate.model_rebuild()
