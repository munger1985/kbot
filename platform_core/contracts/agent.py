"""Main API、Agent Runtime 与独立子 Agent 共享的稳定契约。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CreateAgentRunRequest(BaseModel):
    """公开请求经 Main API 认证后转发的 Run 创建参数。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=0, ge=0, le=999)
    client_metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRunReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    status: str
    event_cursor: int = Field(default=0, ge=0)
    events_url: str


class AgentArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    artifact_type: str
    schema_version: str
    content_hash: str


class AgentRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    agent_id: UUID
    status: str
    row_version: int = Field(ge=1)
    event_cursor: int = Field(ge=0)
    result: AgentArtifactRef | None = None
    error_code: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


class AgentRunEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    task_id: UUID | None = None
    sequence_no: int = Field(ge=1)
    event_type: str
    payload: dict[str, Any]
    created_at: datetime
