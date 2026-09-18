"""规划上下文中的维度值域：治理成员优先，必要时只读观察库存编码。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from loguru import logger

from data_query.connectors.value_members import (
    VALUE_MEMBER_CARDINALITY_LIMIT,
    VALUE_MEMBER_SAMPLE_LIMIT,
    compile_distinct_sample_query,
    members_from_stored_values,
    sample_row_value,
)
from data_query.contracts import (
    DatasetDefinition,
    DimensionDefinition,
    DimensionValueMember,
)


DistinctSampler = Callable[..., Awaitable[tuple[object, ...]]]
VALUE_DOMAIN_CACHE_TTL_SECONDS = 600


def planning_dimension_payload(
    dimension: DimensionDefinition,
    *,
    value_members: tuple[DimensionValueMember, ...] | None = None,
) -> dict[str, object]:
    """只投影逻辑字段和逻辑值域，不暴露物理列。"""
    members = dimension.value_members if value_members is None else value_members
    payload: dict[str, object] = {
        "name": dimension.name,
        "display_name": dimension.display_name,
        "dataset": dimension.dataset,
        "value_type": dimension.value_type,
        "synonyms": dimension.synonyms,
        "groupable": dimension.groupable,
        "filterable": dimension.filterable,
        "allowed_filter_operators": dimension.allowed_filter_operators,
        "value_normalization": dimension.value_normalization,
    }
    if members:
        payload["value_members"] = tuple(
            _planning_member_payload(item) for item in members
        )
    return payload


def _planning_member_payload(member: DimensionValueMember) -> dict[str, object]:
    payload: dict[str, object] = {"value": member.value}
    if member.display_name:
        payload["display_name"] = member.display_name
    if member.aliases:
        payload["aliases"] = member.aliases
    return payload


def should_observe_dimension(dimension: DimensionDefinition) -> bool:
    """只观察可筛选的非敏感短字符串维度。"""
    return (
        not dimension.value_members
        and dimension.value_type == "STRING"
        and dimension.filterable
        and dimension.sensitivity != "SENSITIVE"
        and not dimension.filter_alias_columns
    )


class ValueDomainObserver:
    """对已发布且尚未治理值域的维度做低基数 DISTINCT 观察。"""

    def __init__(
        self,
        *,
        sampler: DistinctSampler | None,
        cache_ttl_seconds: int = VALUE_DOMAIN_CACHE_TTL_SECONDS,
    ) -> None:
        self._sampler = sampler
        self._cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[float, tuple[DimensionValueMember, ...]]] = {}

    async def resolve_planning_members(
        self,
        *,
        dimension: DimensionDefinition,
        dataset: DatasetDefinition,
        source_type: str,
        data_source_id: UUID,
    ) -> tuple[DimensionValueMember, ...]:
        if dimension.value_members:
            return dimension.value_members
        if self._sampler is None or not should_observe_dimension(dimension):
            return ()
        cache_key = (
            f"{data_source_id}:{dataset.physical_schema}:"
            f"{dataset.physical_object}:{dimension.physical_column}"
        )
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]
        try:
            raw_values = await self._sampler(
                source_type=source_type,
                data_source_id=data_source_id,
                schema_name=dataset.physical_schema,
                object_name=dataset.physical_object,
                column_name=dimension.physical_column,
                limit=VALUE_MEMBER_SAMPLE_LIMIT,
            )
            members = members_from_stored_values(raw_values) or ()
            if members and len(members) > VALUE_MEMBER_CARDINALITY_LIMIT:
                members = ()
        except Exception:
            logger.warning(
                "维度值域观察失败，规划将不附带成员 | dimension={}",
                dimension.name,
            )
            return ()
        self._cache[cache_key] = (now + self._cache_ttl_seconds, members)
        logger.info(
            "维度值域观察完成 | dimension={} | member_count={}",
            dimension.name,
            len(members),
        )
        return members


async def sample_distinct_values(
    *,
    executor,
    source_type: str,
    data_source_id: UUID,
    schema_name: str,
    object_name: str,
    column_name: str,
    limit: int,
    query_guardrail: dict[str, int] | None = None,
) -> tuple[object, ...]:
    """通过受治理只读执行器采集库存编码，不把物理名交给 LLM。"""
    compiled = compile_distinct_sample_query(
        dialect=source_type,  # type: ignore[arg-type]
        schema_name=schema_name,
        object_name=object_name,
        column_name=column_name,
        limit=limit,
    )
    budget = {
        "statement_timeout_seconds": 15,
        "max_rows": limit,
        "max_result_bytes": 65_536,
    }
    if query_guardrail:
        budget["statement_timeout_seconds"] = min(
            int(query_guardrail.get("statement_timeout_seconds", 15)), 15,
        )
    result = await executor.execute(
        connector_type=source_type,
        data_source_id=data_source_id,
        query_guardrail=budget,
        compiled=compiled,
    )
    values: list[object] = []
    for row in result.rows:
        if not isinstance(row, dict):
            continue
        values.append(sample_row_value(row))
    return tuple(values)
