"""步骤 4 固定 Blueprint 的 Artifact Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _KernelArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScopeResult(_KernelArtifact):
    schema_version: Literal["SCOPE_RESULT.v1"] = "SCOPE_RESULT.v1"
    target_id: str
    agent_id: str
    trigger_type: str
    target_snapshot: dict[str, Any]
    binding_snapshot: dict[str, Any]
    policy_snapshot: dict[str, Any]


class ObservationSet(_KernelArtifact):
    schema_version: Literal["OBSERVATION_SET.v1"] = "OBSERVATION_SET.v1"
    target_id: str
    observations: tuple[dict[str, Any], ...] = ()
    gaps: tuple[str, ...] = (
        "步骤 4 确定性 Handler 未连接监控或目标数据库",
    )


class KernelReport(_KernelArtifact):
    schema_version: Literal["KERNEL_TEST_REPORT.v1"] = (
        "KERNEL_TEST_REPORT.v1"
    )
    target_id: str
    summary: str
    observation_count: int = Field(ge=0)
    gaps: tuple[str, ...] = ()
