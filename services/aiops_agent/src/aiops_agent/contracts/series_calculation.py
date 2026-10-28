"""可重放的单序列计算；输入只来自同一冻结来源，不扩大监控访问权限。"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SeriesComputation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["SERIES_COMPUTATION.v1"] = "SERIES_COMPUTATION.v1"
    formula_version: Literal["sample-trend-decimal.v1"] = "sample-trend-decimal.v1"
    input_ref: str = Field(pattern=r"^artifact:[^#]+#/observations/[0-9]+/series/[0-9]+/points$")
    row_index: int = Field(ge=0, le=127)
    target_id: str
    source_id: str
    source_version: int = Field(ge=1)
    binding_id: str
    metric_code: str
    unit: str
    value_kind: str
    dimensions: tuple[tuple[str, str], ...] = Field(max_length=128)
    window_start: datetime
    window_end: datetime
    samples: tuple[tuple[datetime, Decimal], ...] = Field(min_length=1, max_length=4096)
    forecast_horizon_days: Decimal | None = Field(default=None, gt=0, le=365)
    summary: dict[str, int | float | str | None] = Field(max_length=32)
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_inputs(self):
        if self.window_start.tzinfo is None or self.window_end.tzinfo is None or self.window_start > self.window_end:
            raise ValueError("序列窗口必须带时区且顺序有效")
        if len(dict(self.dimensions)) != len(self.dimensions):
            raise ValueError("序列维度不能重复")
        if self.forecast_horizon_days is not None and (self.value_kind != "GAUGE" or self.quality_flags):
            raise ValueError("未知语义或不完整序列不能生成预测")
        for stamp, number in self.samples:
            if stamp.tzinfo is None or not self.window_start <= stamp <= self.window_end:
                raise ValueError("序列样本必须处于冻结窗口")
            if not number.is_finite() or len(number.as_tuple().digits) > 1024 or abs(number.as_tuple().exponent) > 1024:
                raise ValueError("序列数值不满足有界精确计算")
        return self
