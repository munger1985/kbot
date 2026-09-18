"""从结构元数据和只读观察结果组装维度值域成员。"""

from __future__ import annotations

import re
from typing import Any, Literal

from data_query.contracts import DimensionValueMember
from data_query.connectors.postgresql import CompiledPostgreSQLQuery


VALUE_MEMBER_CARDINALITY_LIMIT = 64
VALUE_MEMBER_SAMPLE_LIMIT = 65
CLOSED_DOMAIN_FILTER_OPERATORS = ("EQ", "IN", "NE", "NOT_IN", "IS_NULL")
_STRING_TYPE_TOKENS = ("CHAR", "NCHAR", "VARCHAR", "NVARCHAR", "STRING")
_UNSAMPLED_TYPE_TOKENS = ("CLOB", "BLOB", "TEXT", "JSON", "XML", "LONG")
_CHECK_IN_RE = re.compile(
    r'(?:^|[^A-Za-z0-9_])["`]?(?P<column>[A-Za-z_][A-Za-z0-9_$#]*)["`]?\s+IN\s*\((?P<body>.*)\)\s*$',
    re.IGNORECASE | re.DOTALL,
)
_CHECK_ANY_ARRAY_RE = re.compile(
    r'(?:^|[^A-Za-z0-9_])["`]?(?P<column>[A-Za-z_][A-Za-z0-9_$#]*)["`]?'
    r'(?:\s*::\s*[A-Za-z_][A-Za-z0-9_]*)?\s*=\s*ANY\s*\(\s*(?:ARRAY\s*)?\[(?P<body>.*)\]',
    re.IGNORECASE | re.DOTALL,
)
_SQL_STRING_RE = re.compile(r"'(?:''|[^'])*'")


def is_sampleable_string_type(database_type: str) -> bool:
    """只对短字符串类型做 DISTINCT 观察，避免对正文列做高成本采样。"""
    kind = database_type.upper()
    if any(token in kind for token in _UNSAMPLED_TYPE_TOKENS):
        return False
    length = re.search(r"\((\d+)", kind)
    if length and int(length.group(1)) > 128:
        return False
    return any(token in kind for token in _STRING_TYPE_TOKENS)


def quote_identifier(name: str, *, dialect: Literal["POSTGRESQL", "MYSQL", "ORACLE"]) -> str:
    """封装已由治理契约或系统目录给出的标识符。"""
    if dialect == "MYSQL":
        return f"`{name.replace('`', '``')}`"
    return '"' + name.replace('"', '""') + '"'


def compile_distinct_sample_query(
    *,
    dialect: Literal["POSTGRESQL", "MYSQL", "ORACLE"],
    schema_name: str,
    object_name: str,
    column_name: str,
    limit: int = VALUE_MEMBER_SAMPLE_LIMIT,
) -> CompiledPostgreSQLQuery:
    """编译参数化之外的只读 DISTINCT 采样；limit 只能是服务内部常数。"""
    if limit < 1 or limit > VALUE_MEMBER_SAMPLE_LIMIT:
        raise ValueError("VALUE_MEMBER_SAMPLE_LIMIT_INVALID")
    column = quote_identifier(column_name, dialect=dialect)
    source = (
        f"{quote_identifier(schema_name, dialect=dialect)}."
        f"{quote_identifier(object_name, dialect=dialect)}"
    )
    sql = (
        f"SELECT DISTINCT {column} AS {quote_identifier('value', dialect=dialect)} "
        f"FROM {source} WHERE {column} IS NOT NULL ORDER BY 1"
    )
    if dialect == "ORACLE":
        sql = f"{sql} FETCH FIRST {limit} ROWS ONLY"
    else:
        sql = f"{sql} LIMIT {limit}"
    return CompiledPostgreSQLQuery(sql=sql, parameters=())


def members_from_stored_values(values: Any) -> tuple[DimensionValueMember, ...] | None:
    """把库存值变成成员；达到观察上限视为高基数并丢弃。"""
    collected: list[str] = []
    seen: set[str] = set()
    for raw in values or ():
        text = _stored_text(raw)
        if text is None or text in seen:
            continue
        seen.add(text)
        collected.append(text)
        if len(collected) >= VALUE_MEMBER_SAMPLE_LIMIT:
            return None
    if not collected:
        return ()
    return tuple(DimensionValueMember(value=item) for item in collected)


def dimension_value_domain(
    *,
    physical_column: str,
    column_type: str,
    value_samples: Any = (),
    constraints: Any = (),
) -> tuple[tuple[DimensionValueMember, ...], tuple[str, ...] | None]:
    """返回候选值域；CHECK 枚举同时收窄筛选操作符。"""
    checked = members_from_check_constraints(
        physical_column=physical_column, constraints=constraints,
    )
    if checked:
        return checked, CLOSED_DOMAIN_FILTER_OPERATORS
    if not is_sampleable_string_type(column_type):
        return (), None
    sampled = members_from_stored_values(value_samples)
    return sampled or (), None


def members_from_column_metadata(
    *,
    physical_column: str,
    column_type: str,
    value_samples: Any = (),
    constraints: Any = (),
) -> tuple[DimensionValueMember, ...]:
    """优先使用可解析 CHECK 枚举，其次使用低基数采样值。"""
    members, _operators = dimension_value_domain(
        physical_column=physical_column,
        column_type=column_type,
        value_samples=value_samples,
        constraints=constraints,
    )
    return members


def members_from_check_constraints(
    *, physical_column: str, constraints: Any,
) -> tuple[DimensionValueMember, ...]:
    """从 CHECK IN / ANY ARRAY 定义中提取字符串枚举。"""
    if not isinstance(constraints, (list, tuple)):
        return ()
    collected: list[str] = []
    seen: set[str] = set()
    for item in constraints:
        if not isinstance(item, dict):
            continue
        definition = item.get("definition")
        if not isinstance(definition, str) or not definition.strip():
            continue
        values = parse_check_string_members(
            definition=definition, physical_column=physical_column,
        )
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            collected.append(value)
    sampled = members_from_stored_values(collected)
    return sampled or ()


def parse_check_string_members(*, definition: str, physical_column: str) -> tuple[str, ...]:
    """只接受能唯一对应指定列的字符串枚举，解析失败则返回空。"""
    normalized = definition.strip().rstrip(";")
    normalized = re.sub(r"^\s*CHECK\s*\((.*)\)\s*$", r"\1", normalized, flags=re.IGNORECASE | re.DOTALL)
    normalized = normalized.strip()
    match = _CHECK_IN_RE.search(normalized) or _CHECK_ANY_ARRAY_RE.search(normalized)
    if match is None:
        return ()
    if match.group("column").lower() != physical_column.lower():
        return ()
    values = tuple(_SQL_STRING_RE.findall(match.group("body") or ""))
    decoded = []
    seen: set[str] = set()
    for item in values:
        text = item[1:-1].replace("''", "'").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        decoded.append(text)
    return tuple(decoded)


def sample_row_value(row: dict[str, object]) -> object | None:
    """读取 DISTINCT 采样行中的逻辑值列。"""
    for key, value in row.items():
        if str(key).lower() == "value":
            return value
    return next(iter(row.values()), None) if row else None


def _stored_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > 256:
        return None
    return text
