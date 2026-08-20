"""DataQueryPlan.v1：Agent 到 Data Query Service 的受限结构化计划。"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


_KEY_PATTERN = r"^[a-z][a-z0-9._-]{0,127}$"

FilterOperator = Literal[
    "EQ", "NE", "IN", "NOT_IN", "BETWEEN", "GT", "GTE", "LT", "LTE",
    "CONTAINS", "STARTS_WITH", "IS_NULL", "IS_NOT_NULL",
]


class _PlanContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanMeasure(_PlanContract):
    name: str = Field(pattern=_KEY_PATTERN)
    aggregation: Literal["COUNT", "COUNT_DISTINCT", "SUM", "AVG", "MIN", "MAX"]


class PlanFilter(_PlanContract):
    field: str = Field(pattern=_KEY_PATTERN)
    operator: FilterOperator
    values: tuple[str | int | float | bool | None, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_operator_values(self) -> "PlanFilter":
        no_value = {"IS_NULL", "IS_NOT_NULL"}
        collection = {"IN", "NOT_IN"}
        if self.operator in no_value and self.values:
            raise ValueError(f"{self.operator} 不允许 values")
        if self.operator not in no_value and not self.values:
            raise ValueError(f"{self.operator} 必须提供 values")
        if self.operator == "BETWEEN" and len(self.values) != 2:
            raise ValueError("BETWEEN 必须包含两个值")
        if self.operator in collection and len(self.values) > 100:
            raise ValueError(f"{self.operator} 最多 100 个值")
        if (
            self.operator not in collection | {"BETWEEN", "CONTAINS"}
            and len(self.values) != 1
        ):
            raise ValueError(f"{self.operator} 必须包含一个值")
        return self


class PlanOrderBy(_PlanContract):
    field: str = Field(pattern=_KEY_PATTERN)
    direction: Literal["ASC", "DESC"]


class DataQueryPlanV1(_PlanContract):
    contract_version: Literal["DataQueryPlan.v1"] = "DataQueryPlan.v1"
    semantic_model_id: UUID
    semantic_model_version: int = Field(ge=1)
    dataset: str = Field(pattern=_KEY_PATTERN)
    measures: tuple[PlanMeasure, ...] = Field(min_length=1, max_length=32)
    dimensions: tuple[str, ...] = Field(default=(), max_length=32)
    filters: tuple[PlanFilter, ...] = Field(default=(), max_length=64)
    order_by: tuple[PlanOrderBy, ...] = Field(default=(), max_length=8)
    limit: int = Field(default=100, ge=1, le=10000)
    time_zone: str = Field(default="Asia/Shanghai", min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "DataQueryPlanV1":
        measure_names = [item.name for item in self.measures]
        if len(measure_names) != len(set(measure_names)):
            raise ValueError("measures 不能重复")
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError("dimensions 不能重复")
        if any(not re.fullmatch(_KEY_PATTERN, item) for item in self.dimensions):
            raise ValueError("dimensions 包含非法逻辑字段")
        order_fields = [item.field for item in self.order_by]
        if len(order_fields) != len(set(order_fields)):
            raise ValueError("order_by 不能重复")
        return self
