"""Exadata OEM 专用资源合同，不复用数据库 Target 身份。"""

from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ExadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExadataMember(ExadataModel):
    target_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    target_type: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=256)


class ExadataCandidate(ExadataModel):
    candidate_ref: str
    locator_key: str
    display_name: str
    members: tuple[ExadataMember, ...]


class ExadataDiscovery(ExadataModel):
    items: tuple[ExadataCandidate, ...]
    warning: str = "保存所选整机即确认展示的全部成员；成员变化后必须重新发现并确认。"


class ExadataQuery(ExadataModel):
    mode: Literal["LATEST_SNAPSHOT", "HISTORY"] = "LATEST_SNAPSHOT"
    window_minutes: int = Field(default=60, ge=1, le=60)
    scene: Literal["overview", "cell", "disk", "flash", "iorm", "network", "capacity"] = "overview"
    component_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    offset: int = Field(default=0, ge=0, le=10000)
    limit: int = Field(default=12, ge=1, le=24)


class ExadataSample(ExadataModel):
    component_id: str
    metric_code: str
    selector: str
    observed_at: datetime
    value: str | None
    dimensions: dict[str, str]
    unit: str | None
    native_value_type: str
    quality: Literal["OBSERVED", "EMPTY", "STALE", "TRUNCATED"]
    response_sha256: str


class ExadataGap(ExadataModel):
    code: str
    component_id: str | None = None
    metric_code: str | None = None
    detail: str


class ExadataStatusValue(ExadataModel):
    label: str
    tone: Literal["success", "danger", "warning", "neutral", "unknown"]


class ExadataStatusDefinition(ExadataModel):
    category: Literal["AVAILABILITY", "ADMINISTRATIVE", "INDICATOR", "EVENT"]
    display_name: str
    values: dict[str, ExadataStatusValue]
    provenance: str


class ExadataMetricDefinition(ExadataModel):
    code: str
    target_type: str
    selector: str
    display_name: str
    native_value_type: Literal["NUMBER", "STRING"]
    repository_unit_codes: tuple[str, ...]
    keys: tuple[str, ...]
    scene: str
    semantics: str
    aggregation: Literal["NONE"]
    unit_conversion: Literal["NONE"]
    freshness_policy: str
    max_age_seconds: int
    threshold_policy: str
    status: ExadataStatusDefinition | None = None


class ExadataDeviceCoverage(ExadataModel):
    component_id: str
    metric_codes: tuple[str, ...]
    queried_metric_codes: tuple[str, ...]
    note: str


class ExadataView(ExadataModel):
    resource_id: UUID
    binding_id: UUID
    binding_version: int
    source_version: int
    catalog_version: str
    catalog_sha256: str
    captured_at: datetime
    query_mode: Literal["LATEST_SNAPSHOT", "HISTORY"] = "LATEST_SNAPSHOT"
    window_start: datetime | None = None
    window_end: datetime | None = None
    history_resolution: Literal["PROVIDER_MANAGED"] | None = None
    scene: str
    members: tuple[ExadataMember, ...]
    samples: tuple[ExadataSample, ...]
    definitions: tuple[ExadataMetricDefinition, ...]
    device_coverage: tuple[ExadataDeviceCoverage, ...] = ()
    gaps: tuple[ExadataGap, ...]
    partial: bool
    total_queries: int
    next_offset: int | None = None


class ExadataFinding(ExadataModel):
    code: str
    severity: Literal["WARNING", "CRITICAL", "INCONCLUSIVE"]
    component_id: str | None = None
    metric_code: str | None = None
    evidence_indexes: tuple[int, ...] = ()
    detail: str


class ExadataReport(ExadataModel):
    report_id: UUID
    resource_id: UUID
    status: Literal["FINDINGS", "INCONCLUSIVE"]
    findings: tuple[ExadataFinding, ...]
    evidence: ExadataView
    created_at: datetime
