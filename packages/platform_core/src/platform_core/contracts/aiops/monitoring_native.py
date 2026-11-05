"""资源绑定范围内的 OEM 原生监控合同，不暴露来源定位符或查询地址。"""

from datetime import datetime
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


class MonitoringNativeQuery(_Contract):
    metric_name: str = Field(min_length=3, max_length=256)
    window: MonitoringWindowName = "1h"
    mode: Literal["HISTORY", "LATEST"] = "HISTORY"


class MonitoringNativePoint(_Contract):
    observed_at: datetime
    value: str | None
    dimensions: dict[str, str]


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

    @model_validator(mode="after")
    def validate_mode(self):
        if self.query_mode == "LATEST":
            if self.window_start is not None or self.window_end is not None or self.history_resolution != "LATEST_SNAPSHOT":
                raise ValueError("最近采集快照不得声明历史时间窗或历史聚合粒度")
        elif (self.window_start is None or self.window_end is None or self.window_start >= self.window_end
              or self.history_resolution != "PROVIDER_MANAGED"):
            raise ValueError("OEM 历史时序必须保留有效窗口及来源管理的留存粒度")
        return self
