"""精确测量、原生来源及可复核派生计算合同。"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_id: str
    native_origin_ids: tuple[str, ...] = Field(min_length=1)
    physical_identity: tuple[tuple[str, str], ...] = ()
    source_version: str
    content_hash: str = Field(min_length=1)
    collected_start: datetime | None = None
    collected_end: datetime | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_times(self):
        for start, end in ((self.collected_start, self.collected_end), (self.window_start, self.window_end)):
            if (start is None) != (end is None):
                raise ValueError("时间范围必须同时保留起止")
            if start is not None and (
                start.tzinfo is None or end.tzinfo is None or start > end
            ):
                raise ValueError("观测时间必须带时区且顺序有效")
        return self


class ScalarMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_ref: str = Field(min_length=1, max_length=2048)
    series_identity: tuple[tuple[str, str], ...] = Field(min_length=1, max_length=64)
    native_origin_id: str = Field(min_length=1, max_length=2048)
    source_version: str = Field(min_length=1, max_length=4096)
    reset_marker: str | None = None
    observed_at: datetime
    value: Decimal | None
    unit: str = Field(min_length=1, max_length=128)
    kind: Literal["ADDITIVE_COUNTER", "WINDOW_DELTA", "GAUGE", "MAXIMUM", "PERCENTILE", "STATE"]
    window_start: datetime | None = None
    window_end: datetime | None = None
    quality: Literal["GOOD", "MISSING", "PARTIAL", "TRUNCATED", "ESTIMATED", "UNKNOWN"] = "UNKNOWN"
    trust_level: Literal["SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"] = "SOURCE_VERIFIED"

    @model_validator(mode="after")
    def validate_scalar(self):
        if self.unit.strip() != self.unit or any(ord(char) < 32 for char in self.unit):
            raise ValueError("测量单位无效")
        if self.quality == "GOOD" and self.unit.upper() in {"UNKNOWN", "UNSPECIFIED"}:
            raise ValueError("单位未核验的测量不能标记为有效")
        if len(dict(self.series_identity)) != len(self.series_identity):
            raise ValueError("测量身份维度不能重复")
        if self.observed_at.tzinfo is None:
            raise ValueError("测量时间必须带时区")
        if self.kind == "WINDOW_DELTA" and self.window_start is None and self.value is not None:
            raise ValueError("窗口增量必须保留实际覆盖时间")
        if (self.window_start is None) != (self.window_end is None):
            raise ValueError("窗口测量必须同时保留起止时间")
        if self.window_start is not None and (
            self.window_start.tzinfo is None or self.window_end.tzinfo is None
            or self.window_start >= self.window_end or self.observed_at < self.window_end
        ):
            raise ValueError("窗口测量的时间范围无效")
        if self.value is not None and not self.value.is_finite():
            raise ValueError("测量数值必须有限")
        if self.value is not None and (
            len(self.value.as_tuple().digits) > 1024 or abs(self.value.as_tuple().exponent) > 1024
        ):
            raise ValueError("测量精度超出有界计算范围")
        if self.quality == "GOOD" and self.value is None:
            raise ValueError("缺失值不能标记为有效测量")
        return self


class DerivedMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    formula: Literal["AFTER_MINUS_BEFORE", "DELTA_PER_SECOND", "TOTAL_DIVIDED_BY_COUNT"]
    inputs: tuple[ScalarMeasurement, ...] = Field(min_length=2, max_length=2)
    value: Decimal | None
    unit: str
    quality: Literal["GOOD", "PARTIAL", "INCOMPARABLE", "MISSING"]
    reasons: tuple[str, ...] = ()
    native_origin_ids: tuple[str, ...]
