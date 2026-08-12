"""MySQL/Oracle 的受控 Query Plan 编译器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from data_query.contracts import DataQueryPlanV1, SemanticModelDefinition
from data_query.domain import validate_query_plan


@dataclass(frozen=True)
class CompiledDialectQuery:
    sql: str
    parameters: tuple[Any, ...]


def compile_dialect_query(*, dialect: Literal["MYSQL", "ORACLE"], plan: DataQueryPlanV1, model: SemanticModelDefinition, policy_max_limit: int, scope_value: int | None = None) -> CompiledDialectQuery:
    validate_query_plan(plan=plan, model=model, policy_max_limit=policy_max_limit)
    quote = (lambda name: f"`{name}`") if dialect == "MYSQL" else (lambda name: f'"{name}"')
    placeholder = (lambda index: "%s") if dialect == "MYSQL" else (lambda index: f":p{index}")
    dataset = next(item for item in model.datasets if item.name == plan.dataset)
    dimensions = {item.name: item for item in model.dimensions if item.dataset == plan.dataset}
    measures = {item.name: item for item in model.measures if item.dataset == plan.dataset}
    select: list[str] = []; group: list[str] = []; params: list[Any] = []; where: list[str] = []
    if dataset.scope_column is not None:
        if scope_value is None:
            raise ValueError("受 Domain 约束的数据集缺少 scope_value")
        params.append(scope_value)
        where.append(f"{quote(dataset.scope_column)} = {placeholder(len(params))}")
    for name in plan.dimensions:
        column = quote(dimensions[name].physical_column); select.append(f"{column} AS {quote(name)}"); group.append(column)
    for item in plan.measures:
        measure = measures[item.name]
        expression = (
            "COUNT(*)"
            if item.aggregation == "COUNT"
            else f"COUNT(DISTINCT {quote(measure.physical_column or '')})"
            if item.aggregation == "COUNT_DISTINCT"
            else f"{item.aggregation}({quote(measure.physical_column or '')})"
        )
        select.append(f"{expression} AS {quote(item.name)}")
    for item in plan.filters:
        column = quote(dimensions[item.field].physical_column); operator = item.operator
        if operator == "IS_NULL": where.append(f"{column} IS NULL"); continue
        if operator == "IS_NOT_NULL": where.append(f"{column} IS NOT NULL"); continue
        if operator in {"IN", "NOT_IN"}:
            values = item.values; params.extend(values); token = "IN" if operator == "IN" else "NOT IN"
            where.append(f"{column} {token} ({', '.join(placeholder(len(params)-len(values)+offset+1) for offset in range(len(values)))})"); continue
        params.extend(item.values)
        if operator == "BETWEEN": where.append(f"{column} BETWEEN {placeholder(len(params)-1)} AND {placeholder(len(params))}"); continue
        token = placeholder(len(params)); mapping = {"EQ": "=", "NE": "<>", "GT": ">", "GTE": ">=", "LT": "<", "LTE": "<="}
        where.append(f"{column} LIKE ('%' || {token} || '%')" if operator == "CONTAINS" and dialect == "ORACLE" else f"{column} LIKE CONCAT('%', {token}, '%')" if operator == "CONTAINS" else f"{column} LIKE CONCAT({token}, '%')" if operator == "STARTS_WITH" and dialect == "MYSQL" else f"{column} LIKE ({token} || '%')" if operator == "STARTS_WITH" else f"{column} {mapping[operator]} {token}")
    clauses = [f"SELECT {', '.join(select)}", f"FROM {quote(dataset.physical_schema)}.{quote(dataset.physical_object)}"]
    if where: clauses.append("WHERE " + " AND ".join(where))
    if group: clauses.append("GROUP BY " + ", ".join(group))
    if plan.order_by: clauses.append("ORDER BY " + ", ".join(f"{quote(item.field)} {item.direction}" for item in plan.order_by))
    if dialect == "MYSQL": clauses.append("LIMIT %s"); params.append(plan.limit)
    else: clauses.append(f"FETCH FIRST {placeholder(len(params)+1)} ROWS ONLY"); params.append(plan.limit)
    return CompiledDialectQuery("\n".join(clauses), tuple(params))
