"""跨 App Agent 共用的受控图表合同。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ChartContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChartAxis(_ChartContract):
    axis_type: Literal["TIME", "CATEGORY", "VALUE"]
    label: str | None = Field(default=None, max_length=128)
    unit: str | None = Field(default=None, max_length=32)


class ChartPoint(_ChartContract):
    x: str | int | float
    y: float
    display_value: str | None = Field(default=None, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartSeries(_ChartContract):
    series_key: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    unit: str | None = Field(default=None, max_length=32)
    stack: str | None = Field(default=None, max_length=64)
    points: tuple[ChartPoint, ...] = Field(min_length=1, max_length=1000)


class ChartAnnotation(_ChartContract):
    x: str | int | float
    label: str = Field(min_length=1, max_length=256)
    series_key: str | None = Field(default=None, max_length=256)


class ChartSpec(_ChartContract):
    """与 ECharts 等具体前端库解耦的语义图表。"""

    schema_version: Literal["CHART_SPEC.v1"] = "CHART_SPEC.v1"
    chart_type: Literal["LINE", "BAR", "STACKED_BAR"]
    intent: Literal["TREND", "COMPARISON", "CAPACITY"]
    title: str = Field(min_length=1, max_length=256)
    layout: Literal["OVERLAY", "SMALL_MULTIPLES", "NORMALIZED"] = "OVERLAY"
    x_axis: ChartAxis
    y_axis: ChartAxis
    series: tuple[ChartSeries, ...] = Field(min_length=1, max_length=50)
    annotations: tuple[ChartAnnotation, ...] = Field(default=(), max_length=200)
    source_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shape(self) -> "ChartSpec":
        if self.chart_type == "LINE" and self.x_axis.axis_type != "TIME":
            raise ValueError("折线趋势图必须使用 TIME 横轴")
        if self.chart_type == "STACKED_BAR" and not any(
            item.stack for item in self.series
        ):
            raise ValueError("堆叠图至少需要一个 stack")
        return self
