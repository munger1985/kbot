"""从已发布 Semantic Model 编译受限的 PostgreSQL SELECT。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition
from data_query.domain import QueryPlanValidationError, validate_query_plan


@dataclass(frozen=True)
class CompiledPostgreSQLQuery:
    """Connector 私有产物，禁止放入 SSE、Audit payload 或用户 Artifact。"""

    sql: str
    parameters: tuple[Any, ...]


def _quote(identifier: str) -> str:
    """调用者只能传经管理 DTO 验证的标识符，仍统一双引号封装。"""
    return f'"{identifier}"'


def compile_postgresql_query(
    *,
    plan: DataQueryPlanV1,
    model: SemanticModelDefinition,
    policy_max_limit: int,
    scope_value: int | None = None,
) -> CompiledPostgreSQLQuery:
    """编译单条参数化 SELECT；不接受 SQL、函数、Join 或标识符片段输入。"""
    validate_query_plan(plan=plan, model=model, policy_max_limit=policy_max_limit)
    dataset = next(item for item in model.datasets if item.name == plan.dataset)
    dimensions = {item.name: item for item in model.dimensions if item.dataset == plan.dataset}
    measures = {item.name: item for item in model.measures if item.dataset == plan.dataset}

    select_parts: list[str] = []
    group_parts: list[str] = []
    for name in plan.dimensions:
        column = _quote(dimensions[name].physical_column)
        select_parts.append(f"{column} AS {_quote(name)}")
        group_parts.append(column)
    for item in plan.measures:
        definition = measures[item.name]
        if item.aggregation == "COUNT":
            expression = "COUNT(*)"
        else:
            assert definition.physical_column is not None
            expression = f"{item.aggregation}({_quote(definition.physical_column)})"
        select_parts.append(f"{expression} AS {_quote(item.name)}")

    parameters: list[Any] = []
    where_parts: list[str] = []
    if dataset.scope_column is not None:
        if scope_value is None:
            raise ValueError("受 Domain 约束的数据集缺少 scope_value")
        parameters.append(scope_value)
        where_parts.append(f"{_quote(dataset.scope_column)} = ${len(parameters)}")
    for filter_ in plan.filters:
        column = _quote(dimensions[filter_.field].physical_column)
        operator = filter_.operator
        if operator == "IS_NULL":
            where_parts.append(f"{column} IS NULL")
            continue
        if operator == "IS_NOT_NULL":
            where_parts.append(f"{column} IS NOT NULL")
            continue
        if operator in {"IN", "NOT_IN"}:
            placeholders = []
            for value in filter_.values:
                parameters.append(value)
                placeholders.append(f"${len(parameters)}")
            sql_operator = "IN" if operator == "IN" else "NOT IN"
            where_parts.append(f"{column} {sql_operator} ({', '.join(placeholders)})")
            continue
        if operator == "BETWEEN":
            parameters.extend(filter_.values)
            where_parts.append(f"{column} BETWEEN ${len(parameters) - 1} AND ${len(parameters)}")
            continue
        value = filter_.values[0]
        parameters.append(value)
        placeholder = f"${len(parameters)}"
        if operator == "CONTAINS":
            where_parts.append(f"{column} LIKE ('%' || {placeholder} || '%')")
        elif operator == "STARTS_WITH":
            where_parts.append(f"{column} LIKE ({placeholder} || '%')")
        else:
            sql_operator = {"EQ": "=", "NE": "<>", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}[operator]
            where_parts.append(f"{column} {sql_operator} {placeholder}")

    source = f"{_quote(dataset.physical_schema)}.{_quote(dataset.physical_object)}"
    clauses = [f"SELECT {', '.join(select_parts)}", f"FROM {source}"]
    if where_parts:
        clauses.append("WHERE " + " AND ".join(where_parts))
    if group_parts:
        clauses.append("GROUP BY " + ", ".join(group_parts))
    if plan.order_by:
        clauses.append("ORDER BY " + ", ".join(f"{_quote(item.field)} {item.direction}" for item in plan.order_by))
    parameters.append(plan.limit)
    clauses.append(f"LIMIT ${len(parameters)}")
    return CompiledPostgreSQLQuery(sql="\n".join(clauses), parameters=tuple(parameters))
