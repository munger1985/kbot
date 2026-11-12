"""资源绑定范围内的 OEM 原生监控合同，不暴露来源定位符或查询地址。"""

from datetime import datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .monitoring import MonitoringWindowName


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MonitoringNativeMetric(_Contract):
    name: str
    display_name: str
    group: str
    native_value_type: str
    unit: str | None = None
    dimensions: tuple[str, ...] = ()
    query_supported: bool
    latest_supported: bool = False
    current_max_age_seconds: int | None = Field(default=None, gt=0)


class MonitoringNativePanel(_Contract):
    panel_id: str
    title: str
    description: str
    metric: MonitoringNativeMetric | None = None
    mode: Literal["HISTORY", "LATEST"] = "HISTORY"
    unavailable_reason: str | None = None
    baseline_supported: bool = False
    ranking: Literal["ASC", "DESC"] = "DESC"


class MonitoringNativeCatalog(_Contract):
    source_id: UUID
    instance_id: UUID
    entries: tuple[MonitoringNativeMetric, ...]
    default_metric: MonitoringNativeMetric | None = None
    discovered_count: int
    next_offset: int | None = None
    truncated: bool
    revision: str = Field(pattern=r"^[a-f0-9]{64}$")
    warnings: tuple[str, ...] = ()
    overview_panels: tuple[MonitoringNativePanel, ...] = ()


class MonitoringNativeQuery(_Contract):
    metric_name: str = Field(min_length=3, max_length=256)
    window: MonitoringWindowName = "1h"
    mode: Literal["HISTORY", "LATEST"] = "HISTORY"
    baseline: Literal["1w", "1m"] | None = None


class MonitoringNativePoint(_Contract):
    observed_at: datetime
    value: str | None
    dimensions: dict[str, str]
    quality: Literal["OBSERVED", "EMPTY", "STALE", "TRUNCATED"] = "OBSERVED"
    current_eligible: bool = False


class MonitoringNativeBaselineSeries(_Contract):
    dimensions: dict[str, str]
    average: str
    sample_count: int = Field(gt=0)


class MonitoringNativeBaseline(_Contract):
    period: Literal["1w", "1m"]
    start: datetime
    end: datetime
    aggregation: Literal["SAMPLE_MEAN"] = "SAMPLE_MEAN"
    series: tuple[MonitoringNativeBaselineSeries, ...] = ()
    truncated: bool = False
    response_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    unavailable_reason: str | None = None
    note: str = "按各维度的 OEM 留存采样计算算术平均；覆盖率未知，不代表完整时间加权平均。"

    @model_validator(mode="after")
    def validate_baseline(self):
        if self.start.tzinfo is None or self.end.tzinfo is None or self.start >= self.end:
            raise ValueError("历史基线必须包含有效且带时区的窗口")
        if self.series and (self.truncated or self.unavailable_reason):
            raise ValueError("截断或不可用的历史不得生成平均基线")
        try:
            if any(not Decimal(series.average).is_finite() for series in self.series):
                raise ValueError("平均基线必须是有限数值")
        except InvalidOperation as exc:
            raise ValueError("平均基线必须是十进制文本") from exc
        return self


class MonitoringNativeSamples(_Contract):
    source_id: UUID
    instance_id: UUID
    metric: MonitoringNativeMetric
    window_start: datetime | None
    window_end: datetime | None
    captured_at: datetime
    points: tuple[MonitoringNativePoint, ...]
    truncated: bool
    response_sha256: str
    query_mode: Literal["HISTORY", "LATEST"] = "HISTORY"
    history_resolution: Literal["PROVIDER_MANAGED", "LATEST_SNAPSHOT"] = "PROVIDER_MANAGED"
    warnings: tuple[str, ...] = ()
    baseline: MonitoringNativeBaseline | None = None

    @model_validator(mode="after")
    def validate_mode(self):
        if self.query_mode == "LATEST":
            if self.window_start is not None or self.window_end is not None or self.history_resolution != "LATEST_SNAPSHOT":
                raise ValueError("最近采集快照不得声明历史时间窗或历史聚合粒度")
        elif (self.window_start is None or self.window_end is None or self.window_start >= self.window_end
              or self.history_resolution != "PROVIDER_MANAGED"):
            raise ValueError("OEM 历史时序必须保留有效窗口及来源管理的留存粒度")
        return self
