"""Main API、Agent Runtime 与独立子 Agent 共享的稳定契约。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentExecutionSpec(BaseModel):
    """业务 App 签发给 Agent Runtime 的不可变执行规格。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0$")
    owner_app_id: Literal["knowledge_retrieval", "aiops"]
    domain_id: int = Field(ge=1)
    consumer_agent_id: UUID
    consumer_agent_version_id: UUID
    agent_kind: Literal["KNOWLEDGE_RETRIEVAL", "AIOPS"]
    display_name: str = Field(min_length=1, max_length=256)
    enabled_capabilities: tuple[str, ...] = Field(min_length=1)
    models: dict[str, UUID]
    do_rerank: bool = False
    instruction: str | None = Field(default=None, max_length=32000)
    resource_context: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)


class CreateAgentRunRequest(BaseModel):
    """公开请求经 Main API 认证后转发的 Run 创建参数。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    execution_spec: AgentExecutionSpec
    input: str = Field(min_length=1, max_length=32000)
    collection_ids: tuple[UUID, ...] = ()
    security_level: int = Field(default=0, ge=0, le=999)
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_internal_attachment_metadata(self) -> "CreateAgentRunRequest":
        if self.execution_spec.consumer_agent_id != self.agent_id:
            raise ValueError("execution_spec 与 agent_id 不一致")
        if "query_images" in self.client_metadata:
            raise ValueError(
                "查询图片只能通过 Conversation multipart 接口上传"
            )
        return self


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
