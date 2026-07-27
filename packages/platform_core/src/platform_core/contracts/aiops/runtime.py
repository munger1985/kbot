"""AIOps Worker 与运行内核之间的类型化租约契约。"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from .types import (
    AIOpsContract,
    INTERNAL_SCHEMA_VERSION,
    JsonObject,
    OpsRunStatus,
    Sha256Digest,
    UUIDv7,
    UtcDatetime,
)


class ArtifactInput(AIOpsContract):
    artifact_type: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)
    producer: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=64)
    payload: Any | None = None
    payload_uri: str | None = Field(default=None, max_length=2048)
    provenance: JsonObject = Field(default_factory=dict)
    trust_level: str = Field(
        default="SOURCE_VERIFIED", min_length=1, max_length=24
    )
    security_level: int = Field(default=1, ge=0, le=999)

    @model_validator(mode="after")
    def validate_content(self) -> "ArtifactInput":
        if self.payload is None and self.payload_uri is None:
            raise ValueError("Artifact 必须包含 payload 或 payload_uri")
        return self


class LeasedArtifact(AIOpsContract):
    artifact_id: UUIDv7
    artifact_key: str
    artifact_type: str
    schema_version: str
    payload: Any | None = None
    payload_uri: str | None = None
    content_hash: Sha256Digest
    provenance: JsonObject = Field(default_factory=dict)
    trust_level: str
    security_level: int = Field(ge=0, le=999)


class ClaimOpsTaskCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    worker_id: str = Field(min_length=1, max_length=256)
    lease_seconds: int = Field(ge=15, le=3600)
    trace_id: str = Field(min_length=1, max_length=128)


class HeartbeatOpsTaskCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUIDv7
    lease_seconds: int = Field(ge=15, le=3600)


class CompleteOpsTaskCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    artifact: ArtifactInput


class SuspendOpsTaskCommand(AIOpsContract):
    """Worker 请求运行内核原子创建 HITL 并挂起当前 Task。"""

    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUIDv7
    trace_id: str = Field(min_length=1, max_length=128)
    hitl_id: UUIDv7
    request_type: str = Field(min_length=1, max_length=32)
    assignee_user_id: str = Field(min_length=1, max_length=256)
    prompt_text: str = Field(min_length=1, max_length=4000)
    response_schema: JsonObject = Field(default_factory=dict)
    request_artifact: ArtifactInput
    expires_at: UtcDatetime
    idempotency_key: str = Field(min_length=1, max_length=128)


class FailOpsTaskCommand(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUIDv7
    idempotency_key: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    error_code: str = Field(min_length=1, max_length=128)


class TaskLease(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    run_id: UUIDv7
    task_key: str
    task_type: str
    handler_id: str
    handler_version: str
    input_schema_version: str
    output_schema_version: str
    lease_token: UUIDv7
    lease_until: UtcDatetime
    attempt: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    row_version: int = Field(ge=1)
    target_id: UUIDv7
    agent_id: UUIDv7
    actor_id: str
    trace_id: str
    original_request: str
    deadline_at: UtcDatetime | None = None
    plan_snapshot: JsonObject = Field(default_factory=dict)
    policy_snapshot: JsonObject = Field(default_factory=dict)
    input_artifacts: tuple[LeasedArtifact, ...] = ()


class TaskMutationReceipt(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    task_id: UUIDv7
    run_id: UUIDv7
    task_status: str
    run_status: OpsRunStatus
    task_row_version: int = Field(ge=1)
    run_row_version: int = Field(ge=1)
    event_cursor: int = Field(ge=1)
    artifact_id: UUIDv7 | None = None


class OpsRunEventView(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    ops_run_id: UUIDv7
    sequence_no: int = Field(ge=1)
    event_type: str
    visibility: str
    payload: JsonObject
    occurred_at: UtcDatetime


class OpsRunEventPage(AIOpsContract):
    schema_version: str = INTERNAL_SCHEMA_VERSION
    events: tuple[OpsRunEventView, ...] = ()
    next_sequence: int = Field(ge=0)
    terminal: bool = False
