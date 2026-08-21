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


class PlanFilterExpression(_PlanContract):
    """按 filters 下标组合受控谓词，不承载 SQL 片段。"""

    node_type: Literal["FILTER", "ALL", "ANY", "NOT"]
    filter_index: int | None = Field(default=None, ge=0, le=63)
    children: tuple["PlanFilterExpression", ...] = Field(
        default=(),
        max_length=64,
    )
    child: "PlanFilterExpression | None" = None

    @model_validator(mode="after")
    def validate_shape(self) -> "PlanFilterExpression":
        if self.node_type == "FILTER":
            if self.filter_index is None or self.children or self.child is not None:
                raise ValueError("FILTER 只能包含 filter_index")
            return self
        if self.node_type in {"ALL", "ANY"}:
            if len(self.children) < 2 or self.filter_index is not None or self.child is not None:
                raise ValueError("ALL/ANY 必须且只能包含至少两个 children")
            return self
        if self.child is None or self.filter_index is not None or self.children:
            raise ValueError("NOT 必须且只能包含一个 child")
        return self

    def indexes(self) -> tuple[int, ...]:
        if self.node_type == "FILTER":
            return (int(self.filter_index),)
        if self.node_type == "NOT":
            return self.child.indexes() if self.child else ()
        return tuple(
            index for item in self.children for index in item.indexes()
        )

    def depth(self) -> int:
        nested = self.children or ((self.child,) if self.child is not None else ())
        return 1 + max((item.depth() for item in nested), default=0)


class DataQueryPlanV1(_PlanContract):
    contract_version: Literal["DataQueryPlan.v1"] = "DataQueryPlan.v1"
    semantic_model_id: UUID
    semantic_model_version: int = Field(ge=1)
    dataset: str = Field(pattern=_KEY_PATTERN)
    measures: tuple[PlanMeasure, ...] = Field(min_length=1, max_length=32)
    dimensions: tuple[str, ...] = Field(default=(), max_length=32)
    filters: tuple[PlanFilter, ...] = Field(default=(), max_length=64)
    filter_expression: PlanFilterExpression | None = None
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
        if self.filter_expression is not None:
            if self.filter_expression.depth() > 8:
                raise ValueError("filter_expression 深度不能超过 8")
            indexes = self.filter_expression.indexes()
            if any(index >= len(self.filters) for index in indexes):
                raise ValueError("filter_expression 引用了不存在的 filter")
            if set(indexes) != set(range(len(self.filters))):
                raise ValueError("filter_expression 必须引用全部 filters")
        return self


PlanFilterExpression.model_rebuild()
