"""步骤 5 只观测链路的 Artifact Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.contracts.evidence import ObservationSet
from platform_core.contracts.aiops.types import UtcDatetime


class MonitorScopeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["MONITOR_SCOPE_RESULT.v1"] = (
        "MONITOR_SCOPE_RESULT.v1"
    )
    target_id: str
    agent_id: str
    trigger_type: str
    window_start: UtcDatetime
    window_end: UtcDatetime
    catalog_version: str
    catalog_hash: str
    binding_count: int = Field(ge=0)


class MonitorObservationSet(ObservationSet):
    """Artifact 名称与领域 ObservationSet 保持一致。"""


class ObserveReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["OBSERVE_REPORT.v1"] = "OBSERVE_REPORT.v1"
    target_id: str
    status: Literal["READY", "PARTIAL", "FAILED"]
    root_cause_level: Literal["INCONCLUSIVE"] = "INCONCLUSIVE"
    window_start: UtcDatetime
    window_end: UtcDatetime
    source_count: int = Field(ge=0)
    metric_count: int = Field(ge=0)
    alert_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    availability: tuple[dict[str, Any], ...] = ()
    metric_summaries: tuple[dict[str, Any], ...] = ()
    active_alerts: tuple[dict[str, Any], ...] = ()
    gaps: tuple[dict[str, Any], ...] = ()
    evidence_artifact_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
