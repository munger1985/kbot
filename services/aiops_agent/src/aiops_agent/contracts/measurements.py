"""精确测量、原生来源及可复核派生计算合同。"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NativeFieldLineage(BaseModel):
    """固定模板的输出字段到原生计数器的映射，不替代原始数值或窗口证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    column: str = Field(min_length=1, max_length=128)
    native_columns: tuple[str, ...] = Field(min_length=1, max_length=8)
    native_unit: str = Field(min_length=1, max_length=64)
    observed_unit: str = Field(min_length=1, max_length=64)
    kind: Literal["ADDITIVE_COUNTER", "GAUGE", "MAXIMUM", "CUMULATIVE_AVERAGE"]
    unit: str = Field(min_length=1, max_length=64)
    scale_to_canonical: Decimal
    projection: Literal["IDENTITY", "SUM", "PRODUCT", "ROUNDED"] = "IDENTITY"
    native_dimensions: tuple[tuple[str, str], ...] = Field(default=(), max_length=8)
    native_operand_units: tuple[str, ...] = Field(default=(), max_length=8)
    aggregation: Literal["NONE", "SUM", "MAX"] = "NONE"
    estimated: bool = False

    @model_validator(mode="after")
    def validate_scale(self):
        scale = self.scale_to_canonical
        if (not scale.is_finite() or scale <= 0 or len(scale.as_tuple().digits) > 32
                or abs(scale.as_tuple().exponent) > 32):
            raise ValueError("原生单位换算比例无效或超出预算")
        if len(self.native_columns) != len(set(self.native_columns)):
            raise ValueError("原生字段映射不能重复计算同一计数器")
        composite = self.projection in {"SUM", "PRODUCT"}
        if composite and len(self.native_columns) < 2:
            raise ValueError("原生复合投影必须保留多个输入")
        if not composite and len(self.native_columns) != 1:
            raise ValueError("非复合字段必须映射唯一原生输入")
        if self.projection == "PRODUCT":
            if len(self.native_operand_units) != len(self.native_columns):
                raise ValueError("原生乘积必须逐项声明输入单位")
        elif self.native_operand_units:
            raise ValueError("非乘积字段不接受额外输入单位")
        keys = [key for key, _ in self.native_dimensions]
        if len(keys) != len(set(keys)) or any(
            not value or len(value) > 256 or any(ord(char) < 32 for char in value)
            for pair in self.native_dimensions for value in pair
        ):
            raise ValueError("字段原生维度重复、无效或超出预算")
        if any(not unit or len(unit) > 64 or any(ord(char) < 32 for char in unit)
               for unit in self.native_operand_units):
            raise ValueError("原生输入单位无效")
        if self.aggregation == "MAX" and self.kind not in {"MAXIMUM", "GAUGE"}:
            raise ValueError("分组最大值不能声明为可加计数器")
        if self.estimated and self.kind != "GAUGE":
            raise ValueError("估计数值不能声明为精确累计计数器")
        return self


class NativeSeriesScope(BaseModel):
    """原始结果逐行的统计维度；缺失和数据库原生 NULL 保持不同。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_index: int = Field(ge=0)
    dimensions: tuple[tuple[str, str | None], ...] = Field(max_length=16)
    missing_dimensions: tuple[str, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_dimensions(self):
        keys = [key for key, _ in self.dimensions]
        if (len(keys) != len(set(keys)) or len(self.missing_dimensions) != len(set(self.missing_dimensions))
                or set(keys) & set(self.missing_dimensions)):
            raise ValueError("原生统计维度不能重复或同时标记为缺失")
        return self


class NativeMeasurementLineage(BaseModel):
    """模板固定的逐字段血缘；不宣称已解决跨 Exporter 身份或可计算窗口增量。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["NATIVE_MEASUREMENT_LINEAGE.v1"] = "NATIVE_MEASUREMENT_LINEAGE.v1"
    mapping_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    template_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    relation: str = Field(min_length=1, max_length=256)
    fields: tuple[NativeFieldLineage, ...] = Field(min_length=1, max_length=128)
    series: tuple[NativeSeriesScope, ...] = Field(max_length=64)
    reset_scope: Literal["DATABASE_STATISTICS", "STATEMENT_ENTRY", "CURSOR_LIFETIME",
                         "INSTANCE_STATISTICS", "OBJECT_STATISTICS", "PERFORMANCE_SCHEMA_SUMMARY"]
    reset_context_columns: tuple[str, ...] = Field(max_length=8)
    series_truncated: bool
    comparable_window_verified: Literal[False] = False
    exporter_identity_verified: Literal[False] = False

    @model_validator(mode="after")
    def validate_budget(self):
        if len({item.column for item in self.fields}) != len(self.fields):
            raise ValueError("输出字段的原生映射不能重复")
        if tuple(item.row_index for item in self.series) != tuple(range(len(self.series))):
            raise ValueError("原生统计身份必须按原始行顺序保留")
        return self


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
    native_lineage: NativeMeasurementLineage | None = None

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
