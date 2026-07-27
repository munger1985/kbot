"""Skill 与 Runtime Worker 之间的固定契约。"""

from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.application.commands import LeasedArtifact


class ExecutionContext(BaseModel):
    """从持久化 Run/Task 快照构建，不使用进程级可变 Context。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_id: int = Field(ge=1)
    domain_id: int = Field(ge=1)
    agent_id: UUID
    run_id: UUID
    task_id: UUID
    task_key: str
    actor_id: str
    request_id: str
    trace_id: str
    original_input: str
    policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    deadline_at: datetime | None = None
    input_artifacts: tuple[LeasedArtifact, ...] = ()


class SkillArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_type: str = Field(min_length=1, max_length=64)
    schema_version: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] | list[Any] | None = None
    storage_uri: str | None = Field(default=None, max_length=2048)
    provenance: dict[str, Any] = Field(default_factory=dict)
    security_level: int = Field(default=0, ge=0, le=999)
    expires_at: datetime | None = None


class SkillResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact: SkillArtifact
    warnings: tuple[str, ...] = ()


class SkillProgress(BaseModel):
    """Skill 在最终 Artifact 前持久化的可恢复增量事件。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeSkill(Protocol):
    async def execute(self, context: ExecutionContext) -> SkillResult: ...
