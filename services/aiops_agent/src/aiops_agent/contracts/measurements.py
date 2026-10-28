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

    input_ref: str
    series_identity: tuple[tuple[str, str], ...] = Field(min_length=1)
    native_origin_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    reset_marker: str | None = None
    observed_at: datetime
    value: Decimal | None
    unit: str
    kind: Literal["ADDITIVE_COUNTER", "GAUGE", "MAXIMUM", "PERCENTILE", "STATE"]
    quality: Literal["GOOD", "MISSING", "PARTIAL", "TRUNCATED", "ESTIMATED", "UNKNOWN"] = "UNKNOWN"
    trust_level: Literal["SOURCE_VERIFIED", "USER_PROVIDED", "MODEL_INFERENCE"] = "SOURCE_VERIFIED"

    @model_validator(mode="after")
    def validate_scalar(self):
        if self.observed_at.tzinfo is None:
            raise ValueError("测量时间必须带时区")
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
