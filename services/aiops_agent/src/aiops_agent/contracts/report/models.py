"""REPORT_CONTENT.v1 严格 Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts.aiops.types import UtcDatetime


class ReportContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["REPORT_CONTENT.v1"] = "REPORT_CONTENT.v1"
    report_key: str
    report_type: Literal[
        "INCIDENT",
        "PERFORMANCE",
        "INSPECTION_DAILY",
        "INSPECTION_WEEKLY",
        "INSPECTION_CUSTOM",
        "COMPARISON",
    ]
    ops_run_id: str
    target_id: str
    title: str
    status: Literal["READY", "PARTIAL"]
    summary: str
    period_start: UtcDatetime
    period_end: UtcDatetime
    scope: dict[str, Any]
    facts: tuple[dict[str, Any], ...] = ()
    gaps: tuple[dict[str, Any], ...] = ()
    evidence_refs: tuple[dict[str, str], ...] = ()
    recommendations: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)
