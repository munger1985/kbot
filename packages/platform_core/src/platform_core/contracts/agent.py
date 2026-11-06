"""Main API、Agent Runtime 与独立子 Agent 共享的稳定契约。"""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

class KnowledgeLibraryBinding(BaseModel):
    """仅绑定所属 App 资料库；Run 建立后由 App 重新鉴权并冻结精确版本。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    library_id: UUID | None = None
    query_ticket: UUID | None = None
    profile_ref: str = Field(default="default", min_length=1, max_length=128)

    @model_validator(mode="after")
    def exact_binding(self):
        if (self.library_id is None) == (self.query_ticket is None):
            raise ValueError("必须且只能指定资料库或 AIOps 查询委托")
        return self


class AgentExecutionSpec(BaseModel):
    """业务 App 签发给 Agent Runtime 的不可变执行规格。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(pattern=r"^1\.0$")
    owner_app_id: Literal["knowledge_retrieval", "aiops"]
    domain_id: int = Field(ge=1)
    consumer_agent_id: UUID
    consumer_agent_version_id: UUID
    agent_kind: Literal["KNOWLEDGE_RETRIEVAL", "AIOPS", "AIOPS_KNOWLEDGE"]
    display_name: str = Field(min_length=1, max_length=256)
    enabled_capabilities: tuple[
        Literal["conversation", "document", "data_query", "aiops"], ...
    ] = Field(min_length=1)
    models: dict[str, UUID]
    instruction: str | None = Field(default=None, max_length=32000)
    resource_context: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    knowledge_binding: KnowledgeLibraryBinding | None = None

    @model_validator(mode="after")
    def validate_agent_boundary(self) -> "AgentExecutionSpec":
        """禁止知识检索与 AIOps Agent 混用所有权和能力。"""
        capabilities = set(self.enabled_capabilities)
        if len(capabilities) != len(self.enabled_capabilities):
            raise ValueError("enabled_capabilities 不能包含重复能力")
        if self.agent_kind == "AIOPS_KNOWLEDGE":
            if (self.owner_app_id != "aiops" or capabilities != {"document"}
                or self.knowledge_binding is None or self.knowledge_binding.query_ticket is None):
                raise ValueError("AIOps 知识子执行只能读取本 App 的精确查询委托")
            return self
        if self.agent_kind == "KNOWLEDGE_RETRIEVAL":
            if self.owner_app_id != "knowledge_retrieval" or (
                self.knowledge_binding is not None and self.knowledge_binding.library_id is None):
                raise ValueError("知识检索 Agent 必须绑定自身 App 资料库")
            if not capabilities.issubset(
                {"conversation", "document", "data_query"}
            ):
                raise ValueError("知识检索 Agent 不能启用 AIOps 能力")
            if ("document" in capabilities) != (self.knowledge_binding is not None):
                raise ValueError("Document 能力与资料库绑定必须同时配置")
        elif self.owner_app_id != "aiops" or capabilities != {"aiops"}:
            raise ValueError("AIOps Agent 必须由 aiops 拥有且只能启用 aiops")
        elif self.knowledge_binding is not None:
            raise ValueError("AIOps 诊断 Agent 不能携带知识问答绑定")
        return self


class CreateAgentRunRequest(BaseModel):
    """公开请求经 Main API 认证后转发的 Run 创建参数。"""

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    execution_spec: AgentExecutionSpec
    input: str = Field(min_length=1, max_length=32000)
    client_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_internal_attachment_metadata(self) -> "CreateAgentRunRequest":
        if self.execution_spec.consumer_agent_id != self.agent_id:
            raise ValueError("execution_spec 与 agent_id 不一致")
        if self.execution_spec.agent_kind != "KNOWLEDGE_RETRIEVAL":
            raise ValueError("Agent Runtime 聊天运行只接受知识检索 Agent")
        if "query_images" in self.client_metadata:
            raise ValueError(
                "查询图片只能通过 Conversation Turn 的 images 字段上传"
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
    owner_app_id: str
    scope_hash: str | None = None
    status: str
    row_version: int = Field(ge=1)
    event_cursor: int = Field(ge=0)
    result: AgentArtifactRef | None = None
    error_code: str | None = None
    error_message: str | None = None
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

class CreateAIOpsKnowledgeRunRequest(BaseModel):
    """仅 AIOps 可信内部调用方可创建；不创建业务 Agent 或独立会话。"""
    model_config = ConfigDict(extra="forbid")
    child_run_id: UUID
    parent_turn_id: UUID
    parent_ops_run_id: UUID | None = None
    execution_spec: AgentExecutionSpec
    input: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def child_only(self):
        if self.execution_spec.agent_kind != "AIOPS_KNOWLEDGE":
            raise ValueError("该接口仅接受 AIOps 知识子执行")
        return self
