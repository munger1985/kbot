"""AIOps SSE 与跨服务安全事件契约。"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import ConfigDict, Field

from .types import (
    AIOpsContract,
    ArtifactRef,
    EVENT_SCHEMA_VERSION,
    ExecutionStatus,
    HitlType,
    OpsRunStatus,
    ProposalStatus,
    ReportStatus,
    ReportType,
    ResourceRef,
    UUIDv7,
    UtcDatetime,
)


class EventBase(AIOpsContract):
    schema_version: str = EVENT_SCHEMA_VERSION
    ops_run_id: UUIDv7
    sequence_no: int = Field(ge=1)
    occurred_at: UtcDatetime
    trace_id: str = Field(min_length=1, max_length=128)


class RunStatusEvent(EventBase):
    event_type: Literal["run.status"] = "run.status"
    status: OpsRunStatus


class TaskStatusEvent(EventBase):
    event_type: Literal["task.status"] = "task.status"
    status: str
    task_id: UUIDv7
    task_type: str


class DiagnosticProgressEvent(EventBase):
    event_type: Literal["diagnostic.progress"] = "diagnostic.progress"
    status: str
    stage: str
    progress_percent: int | None = Field(default=None, ge=0, le=100)


class InputRequiredEvent(EventBase):
    event_type: Literal["diagnostic.input_required"] = (
        "diagnostic.input_required"
    )
    status: Literal["PENDING"] = "PENDING"
    hitl_id: UUIDv7
    hitl_type: HitlType
    request_artifact: ArtifactRef
    expires_at: UtcDatetime


class ApprovalRequiredEvent(EventBase):
    event_type: Literal["proposal.pending_approval"] = (
        "proposal.pending_approval"
    )
    status: ProposalStatus
    proposal_id: UUIDv7
    proposal_hash: str


class ExecutionStatusEvent(EventBase):
    event_type: Literal["execution.status"] = "execution.status"
    status: ExecutionStatus
    execution_id: UUIDv7
    result_ref: ResourceRef | None = None


class ReportReadyEvent(EventBase):
    event_type: Literal["report.ready"] = "report.ready"
    status: ReportStatus
    report_id: UUIDv7
    report_key: str
    report_type: ReportType
    report_version: int = Field(ge=1)
    summary: str


class RunTerminalEvent(EventBase):
    event_type: Literal[
        "run.completed",
        "run.failed",
        "run.cancelled",
        "run.expired",
    ]
    status: Literal["COMPLETED", "FAILED", "CANCELLED", "EXPIRED"]
    final_artifact: ArtifactRef | None = None
    error_code: str | None = None


class UnknownEvent(EventBase):
    """客户端可记录并跳过的前向兼容事件。"""

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        use_enum_values=True,
    )

    event_type: str
    status: str | None = None


AIOpsEvent = Annotated[
    Union[
        RunStatusEvent,
        TaskStatusEvent,
        DiagnosticProgressEvent,
        InputRequiredEvent,
        ApprovalRequiredEvent,
        ExecutionStatusEvent,
        ReportReadyEvent,
        RunTerminalEvent,
    ],
    Field(discriminator="event_type"),
]

SafeAIOpsEvent = Union[AIOpsEvent, UnknownEvent]
