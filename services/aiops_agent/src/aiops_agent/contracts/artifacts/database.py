"""步骤 6 数据库诊断 Artifact Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts.aiops.executor import DatabaseObservation


class DatabaseScopeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DATABASE_SCOPE_RESULT.v1"] = (
        "DATABASE_SCOPE_RESULT.v1"
    )
    target_id: str
    db_type: Literal["POSTGRESQL", "ORACLE", "MYSQL"]
    configured_version: str
    catalog_hash: str
    capability_snapshot_hash: str
    selected_tool_count: int = Field(ge=0)
    initial_gaps: tuple[dict[str, Any], ...] = ()


class EvidenceGap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    tool_id: str | None = None
    detail: str
    retryable: bool = False


class DatabaseDiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DATABASE_DIAGNOSTIC_RESULT.v1"] = (
        "DATABASE_DIAGNOSTIC_RESULT.v1"
    )
    target_id: str
    tool_id: str
    status: Literal["SUCCEEDED", "GAP"]
    observation: DatabaseObservation | None = None
    gap: EvidenceGap | None = None


class DatabaseObservationAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DATABASE_OBSERVATION_AGGREGATE.v1"] = (
        "DATABASE_OBSERVATION_AGGREGATE.v1"
    )
    target_id: str
    observations: tuple[DatabaseObservation, ...] = ()
    gaps: tuple[EvidenceGap, ...] = ()


class DatabaseDiagnosticReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["DB_DIAGNOSTIC_REPORT.v1"] = (
        "DB_DIAGNOSTIC_REPORT.v1"
    )
    target_id: str
    status: Literal["READY", "PARTIAL"]
    root_cause_level: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"
    observation_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    tools: tuple[str, ...] = ()
    gaps: tuple[EvidenceGap, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
