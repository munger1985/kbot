"""Main API、Agent Runtime 与独立子 Agent 共享的稳定契约。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CreateAgentRunRequest(BaseModel):
    """公开请求经 Main API 认证后转发的 Run 创建参数。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=0, ge=0, le=999)
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_internal_attachment_metadata(self) -> "CreateAgentRunRequest":
        if "query_images" in self.client_metadata:
            raise ValueError(
                "查询图片只能通过 Conversation multipart 接口上传"
            )
        return self


class CreateAgentDefinitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_key: str = Field(pattern=r"^[a-z][a-z0-9._-]{0,127}$")
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] = Field(min_length=1)
    models: dict[str, UUID]
    do_rerank: bool = False
    data_profile_name: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="DRAFT", pattern=r"^(DRAFT|ACTIVE)$")


class UpdateAgentDefinitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    enabled_capabilities: tuple[str, ...] | None = None
    models: dict[str, UUID] | None = None
    do_rerank: bool | None = None
    data_profile_name: str | None = Field(
        default=None, min_length=1, max_length=256
    )
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: str | None = Field(
        default=None, pattern=r"^(DRAFT|ACTIVE|INACTIVE)$"
    )


class AgentDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: UUID
    domain_id: int
    agent_key: str
    display_name: str
    description: str | None = None
    status: str
    enabled_capabilities: tuple[str, ...]
    models: dict[str, UUID]
    do_rerank: bool
    data_profile_name: str | None = None
    instruction: str | None = None
    config: dict[str, Any]
    row_version: int = Field(ge=1)


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


class AgentArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: UUID
    artifact_type: str
    schema_version: str
    producer: str
    producer_version: str
    payload: dict[str, Any] | list[Any] | None = None
    storage_uri: str | None = None
    content_hash: str
    provenance: dict[str, Any]
    security_level: int
    created_at: datetime


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
