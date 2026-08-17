"""Main API 与 Agent Runtime 共享的 Conversation/Memory 契约。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentExecutionSpec


class _Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateConversationRequest(_Contract):
    agent_id: UUID
    execution_spec: AgentExecutionSpec
    title: str | None = Field(default=None, min_length=1, max_length=512)
    retention_policy: str = Field(
        default="DEFAULT",
        pattern=r"^(DEFAULT|KEEP_FOREVER|DAYS_30|DAYS_90|DAYS_365)$",
    )


class ConversationView(_Contract):
    conversation_id: UUID
    agent_id: UUID
    title: str | None = None
    status: str
    row_version: int = Field(ge=1)
    last_turn_sequence: int = Field(ge=0)
    last_active_at: datetime
    created_at: datetime
    retention_policy: str
    purge_after: datetime | None = None


class UpdateConversationRequest(_Contract):
    expected_row_version: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=512)
    status: str | None = Field(
        default=None, pattern=r"^(ACTIVE|ARCHIVED)$"
    )
    retention_policy: str | None = Field(
        default=None,
        pattern=r"^(DEFAULT|KEEP_FOREVER|DAYS_30|DAYS_90|DAYS_365)$",
    )


class CreateConversationTurnRequest(_Contract):
    input: str = Field(min_length=1, max_length=32000)
    expected_conversation_version: int = Field(ge=1)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=3, ge=0, le=3)
    client_metadata: dict[str, Any] = Field(default_factory=dict)
    images: tuple["ConversationQueryImage", ...] = Field(
        default=(), max_length=8
    )


class ConversationQueryImage(_Contract):
    file_name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(pattern=r"^image/(png|jpeg|webp)$")
    content_base64: str = Field(min_length=1, max_length=24 * 1024 * 1024)


class ConversationTurnReceipt(_Contract):
    conversation_id: UUID
    turn_id: UUID
    turn_sequence: int = Field(ge=1)
    turn_status: str
    run_id: UUID | None = None
    run_status: str | None = None
    event_cursor: int = Field(default=0, ge=0)
    events_url: str | None = None


class ConversationItemView(_Contract):
    item_id: UUID
    item_sequence: int = Field(ge=1)
    item_type: str
    role: str
    content: dict[str, Any]
    run_id: UUID | None = None
    artifact_id: UUID | None = None
    created_at: datetime


class PublicTraceEvent(_Contract):
    schema_version: str = "AgentTraceEvent.v1"
    run_id: UUID
    turn_id: UUID
    task_id: UUID | None = None
    sequence_no: int = Field(ge=1)
    stage: str
    title: str
    summary: str
    status: str
    resource_refs: tuple[dict[str, Any], ...] = ()
    occurred_at: datetime


class ConversationTurnView(_Contract):
    conversation_id: UUID
    turn_id: UUID
    turn_sequence: int = Field(ge=1)
    status: str
    run_id: UUID | None = None
    user_item: ConversationItemView | None = None
    assistant_item: ConversationItemView | None = None
    trace_summary: tuple[PublicTraceEvent, ...] = ()
    created_at: datetime
    completed_at: datetime | None = None
    memory_status: str | None = None


class ConversationTurnPage(_Contract):
    conversation_id: UUID
    turns: tuple[ConversationTurnView, ...]
    next_sequence: int = Field(ge=0)


class MemoryItemView(_Contract):
    memory_id: UUID
    agent_id: UUID | None = None
    memory_type: str
    scope_type: str
    canonical_key: str
    value: dict[str, Any]
    confidence: float = Field(ge=0, le=1)
    salience: float = Field(ge=0, le=1)
    valid_from: datetime
    valid_to: datetime | None = None
    status: str
    created_at: datetime
    updated_at: datetime
