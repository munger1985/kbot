"""按语义维度契约规范化筛选参数。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from data_query.contracts import DimensionDefinition


def normalize_filter_values(
    *, dimension: DimensionDefinition, values: tuple[Any, ...]
) -> tuple[Any, ...]:
    """在参数绑定前完成确定性的文本和日期类型规范化。"""
    return tuple(_normalize_filter_value(dimension=dimension, value=value) for value in values)


def _normalize_filter_value(*, dimension: DimensionDefinition, value: Any) -> Any:
    if value is None:
        return None
    if dimension.value_type == "DATE":
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip())
            except ValueError as exc:
                raise ValueError("DATE_FILTER_VALUE_INVALID") from exc
        raise ValueError("DATE_FILTER_VALUE_INVALID")
    if dimension.value_type == "DATETIME":
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(normalized)
            except ValueError as exc:
                raise ValueError("DATETIME_FILTER_VALUE_INVALID") from exc
        raise ValueError("DATETIME_FILTER_VALUE_INVALID")
    if dimension.value_normalization == "LOWER_TRIM" and isinstance(value, str):
        return value.strip().lower()
    return value
