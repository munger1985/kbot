"""ASM、RAC、Listener 的跨监控源显式映射，不接受查询语言。"""

from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ORACLE_SERVICE_TYPES = frozenset({"ORACLE_ASM", "ORACLE_RAC", "ORACLE_LISTENER"})
OEM_SERVICE_TYPES = {"ORACLE_ASM": {"osm_cluster", "osm_instance"},
    "ORACLE_RAC": {"rac_database"}, "ORACLE_LISTENER": {"oracle_listener"}}


def state_key(value, native_value_type):
    """数字状态用精确十进制比较；文本保持大小写，不做浮点近似。"""
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    if native_value_type != 'NUMBER':
        return value
    try:
        numeric = Decimal(value)
        return numeric if numeric.is_finite() else None
    except InvalidOperation:
        return None


class OracleServiceState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str = Field(min_length=1, max_length=64)
    tone: Literal["success", "danger", "warning", "neutral", "unknown"]


class OracleServiceMetricMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    metric_name: str = Field(min_length=1, max_length=256)
    native_unit: str | None = Field(default=None, max_length=64)
    native_value_type: Literal["NUMBER", "STRING"] = "NUMBER"
    dimensions: dict[str, str] = Field(default_factory=dict, max_length=16)
    fixed_dimensions: dict[str, str] = Field(default_factory=dict, max_length=16)
    scale: str = Field(default="1", min_length=1, max_length=64)
    state_values: dict[str, OracleServiceState] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def validate_mapping(self):
        try:
            number = Decimal(self.scale)
            if not number.is_finite() or not Decimal('1e-12') <= number <= Decimal('1e12'):
                raise ValueError
        except (InvalidOperation, ValueError):
            raise ValueError("单位比例必须为有限正十进制数") from None
        if self.native_value_type != "NUMBER" and number != 1:
            raise ValueError("文本原值不可数值换算")
        if len(set(self.dimensions.values())) != len(self.dimensions) or set(self.dimensions.values()) & set(self.fixed_dimensions):
            raise ValueError("维度映射不能重复或覆盖固定维度")
        for key, value in (*self.dimensions.items(), *self.fixed_dimensions.items()):
            if not key or not value or len(key) > 64 or len(value) > 256 or any(ord(c) < 32 for c in key + value):
                raise ValueError("维度名称或值无效")
        if any(ord(c) < 32 for c in self.metric_name):
            raise ValueError("原生指标名称不可包含控制字符")
        keys = [state_key(key, self.native_value_type) for key in self.state_values]
        if None in keys or len(set(keys)) != len(keys):
            raise ValueError('状态字典不可包含无效或等价重复的原始值')
        return self
