"""Agent Runtime 内部命令与最小回执。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_runtime.domain.planning import PlanDraft


class _FrozenCommand(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateRunCommand(_FrozenCommand):
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    agent_id: UUID
    actor_id: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)
    original_input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=0, ge=0, le=999)
    client_metadata: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: UUID | None = None
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    deadline_at: datetime | None = None
    conversation_id: UUID | None = None
    turn_id: UUID | None = None
    conversation_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_conversation_scope(self) -> "CreateRunCommand":
        if (self.conversation_id is None) != (self.turn_id is None):
            raise ValueError(
                "conversation_id 与 turn_id 必须同时提供或同时为空"
            )
        if self.conversation_id is None and self.conversation_context:
            raise ValueError("无 Conversation 的 Run 不能携带会话上下文")
        if (
            self.conversation_id is None
            and "query_images" in self.client_metadata
        ):
            raise ValueError(
                "无 Conversation 的 Run 不能携带查询图片存储描述"
            )
        return self


class InstallPlanCommand(_FrozenCommand):
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    run_id: UUID
    expected_row_version: int = Field(ge=1)
    plan: PlanDraft
    actor_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class ClaimTaskCommand(_FrozenCommand):
    worker_id: str = Field(min_length=1, max_length=256)
    lease_seconds: int = Field(ge=15, le=3600)
    trace_id: str = Field(min_length=1, max_length=128)


class HeartbeatTaskCommand(_FrozenCommand):
    task_id: UUID
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    lease_seconds: int = Field(ge=15, le=3600)


class ArtifactInput(_FrozenCommand):
    artifact_type: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)
    producer: str = Field(min_length=1, max_length=128)
    producer_version: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] | list[Any] | None = None
    storage_uri: str | None = Field(default=None, max_length=2048)
    provenance: dict[str, Any] = Field(default_factory=dict)
    security_level: int = Field(default=0, ge=0, le=999)
    expires_at: datetime | None = None


class CompleteTaskCommand(_FrozenCommand):
    task_id: UUID
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    artifact: ArtifactInput
    actor_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class AppendTaskProgressCommand(_FrozenCommand):
    task_id: UUID
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    event_type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)
    actor_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class FailTaskCommand(_FrozenCommand):
    task_id: UUID
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    error_code: str = Field(min_length=1, max_length=128)
    error_message: str = Field(min_length=1, max_length=1000)
    retryable: bool = False
    retry_at: datetime | None = None
    actor_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_retry(self) -> "FailTaskCommand":
        if self.retryable and self.retry_at is None:
            raise ValueError("可重试失败必须提供 retry_at")
        return self


class StartDelegationCommand(_FrozenCommand):
    task_id: UUID
    expected_row_version: int = Field(ge=1)
    worker_id: str = Field(min_length=1, max_length=256)
    lease_token: UUID
    trace_id: str = Field(min_length=1, max_length=128)


class CancelRunCommand(_FrozenCommand):
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    run_id: UUID
    expected_row_version: int = Field(ge=1)
    actor_id: str = Field(min_length=1, max_length=256)
    trace_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=128)


class TaskLease(_FrozenCommand):
    task_id: UUID
    run_id: UUID
    task_key: str
    task_type: str
    row_version: int
    lease_token: UUID
    lease_until: datetime
    attempt: int
    timeout_seconds: int
    execution_kind: str
    specialist: str | None = None
    skill_id: str | None = None
    skill_version: str | None = None
    delegate_service: str | None = None
    delegate_capability: str | None = None
    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    agent_id: UUID
    actor_id: str
    request_id: str
    trace_id: str
    original_input: str
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    deadline_at: datetime | None = None
    input_refs: tuple[str, ...] = ()
    input_artifacts: tuple["LeasedArtifact", ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()


class LeasedArtifact(_FrozenCommand):
    artifact_id: UUID
    task_id: UUID | None = None
    artifact_type: str
    schema_version: str
    producer: str
    producer_version: str
    payload: dict[str, Any] | list[Any] | None = None
    storage_uri: str | None = None
    content_hash: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    security_level: int = Field(ge=0, le=999)


class TaskMutationReceipt(_FrozenCommand):
    task_id: UUID
    run_id: UUID
    task_status: str
    task_row_version: int
    run_status: str
    run_row_version: int
    event_cursor: int = Field(ge=1)
    artifact_id: UUID | None = None
