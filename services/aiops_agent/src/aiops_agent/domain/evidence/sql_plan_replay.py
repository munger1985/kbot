"""估算专用SQL与原生参数的有界编译；不改写原业务查询的LIMIT或访问条件。"""

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import re

from sqlglot import Dialect, exp
from sqlglot.errors import ErrorLevel, SqlglotError
from sqlglot.optimizer.scope import traverse_scope

from aiops_agent.domain.evidence.sql_bindings import sql_bind_positions
from aiops_agent.domain.evidence.sql_plan_capture import inspect_plan_capture


class SqlPlanReplayRejected(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class SqlPlanReplay:
    sql: str = field(repr=False)
    parameters: tuple = field(repr=False)
    native_types: tuple[str, ...]


# 只列已实现的内置函数族；不把用户函数或包函数因名字近似而当成纯函数。
_SAFE_FUNCTIONS = frozenset({
    "ABS", "AVG", "CAST", "CEIL", "COALESCE", "COUNT", "DATE", "DATE_TRUNC", "TIMESTAMP_TRUNC",
    "EXTRACT", "FLOOR", "GREATEST", "LEAST", "LENGTH", "LOWER", "MAX", "MIN", "NULLIF", "ROUND",
    "SUM", "TRIM", "UPPER", "SUBSTRING", "STR_POSITION", "STDDEV", "VARIANCE",
})
_STRINGS = {
    "POSTGRESQL": {"text", "varchar", "bpchar", "name", "uuid"},
    "MYSQL": {"char", "varchar", "text", "tinytext", "mediumtext", "longtext"},
    "ORACLE": {"varchar2", "nvarchar2", "char", "nchar"},
}
_INTEGERS = {"POSTGRESQL": {"int2", "int4", "int8"}, "MYSQL": {"tinyint", "smallint", "mediumint", "int", "bigint"}, "ORACLE": set()}
_DECIMALS = {"POSTGRESQL": {"numeric"}, "MYSQL": {"decimal", "numeric"}, "ORACLE": {"number"}}
_TEMPORALS = {"POSTGRESQL": {"timestamp", "timestamptz", "date", "time"},
    "MYSQL": {"timestamp", "datetime", "date", "time"}, "ORACLE": {"date", "timestamp"}}
_ALIASES = {
    "POSTGRESQL": {"smallint": "int2", "integer": "int4", "int": "int4", "bigint": "int8",
        "decimal": "numeric", "boolean": "bool", "character varying": "varchar", "character": "bpchar",
        "timestamp without time zone": "timestamp", "timestamp with time zone": "timestamptz",
        "time without time zone": "time"},
    "MYSQL": {"integer": "int", "bool": "boolean", "dec": "decimal"}, "ORACLE": {},
}
_INTEGER_BITS = {"int2": 16, "int4": 32, "int8": 64, "tinyint": 8, "smallint": 16,
    "mediumint": 24, "int": 32, "bigint": 64}


def _native_type(original, database_type):
    # 只规范明确的内置别名；不丢弃长度、精度、unsigned、数组或用户类型限定。
    name = " ".join(original.lower().split())
    if database_type == "POSTGRESQL":
        name = name.removeprefix("pg_catalog.")
    name = _ALIASES[database_type].get(name, name)
    supported = _STRINGS[database_type] | _INTEGERS[database_type] | _DECIMALS[database_type] | _TEMPORALS[database_type]
    if database_type != "ORACLE":
        supported = supported | ({"bool"} if database_type == "POSTGRESQL" else {"boolean"})
    if name not in supported:
        raise SqlPlanReplayRejected("PLAN_NATIVE_TYPE_UNSUPPORTED")
    return name


def _decode(value, name, database_type):
    if value is None:
        return None
    if name in _STRINGS[database_type]:
        if not isinstance(value, str):
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        if name == "uuid":
            from uuid import UUID
            try:
                return UUID(value)
            except ValueError:
                raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH") from None
        return value
    if name in _INTEGERS[database_type]:
        if not isinstance(value, str) or not re.fullmatch(r"-?(?:0|[1-9]\d{0,19})", value):
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        number = int(value)
        bound = 2 ** (_INTEGER_BITS[name] - 1)
        if not -bound <= number < bound:
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        return number
    if name in _DECIMALS[database_type]:
        if not isinstance(value, str) or not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", value):
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        number = Decimal(value)
        if not number.is_finite() or abs(number.adjusted()) > 1000:
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        return number
    if name in {"bool", "boolean"} and database_type != "ORACLE":
        if not isinstance(value, bool):
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        return value
    if name in _TEMPORALS[database_type]:
        if not isinstance(value, str):
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        if name == "date":
            if database_type != "ORACLE":
                return date.fromisoformat(value)
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
            return parsed
        if name == "time":
            parsed = time.fromisoformat(value)
            if parsed.tzinfo is not None:
                raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
            return parsed
        stamp = datetime.fromisoformat(value)
        if name == "timestamptz" and stamp.tzinfo is None or name in {"timestamp", "datetime"} and stamp.tzinfo is not None:
            raise SqlPlanReplayRejected("PLAN_PARAMETER_TYPE_MISMATCH")
        return stamp
    raise SqlPlanReplayRejected("PLAN_NATIVE_TYPE_UNSUPPORTED")


def compile_sql_plan_replay(*, database_type, sql, document, allowed_objects):
    """输入仍来自保护通道；所有异常仅暴露固定原因码，不回显SQL、类型或原值。"""
    try:
        parameters = document["parameters"]
        inspected = inspect_plan_capture(database_type=database_type, sql=sql, parameter_count=len(parameters))
        if inspected.status != "AWAITING_DATABASE_VALIDATION":
            raise SqlPlanReplayRejected("PLAN_SOURCE_NOT_REPLAYABLE")
        dialect = {"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type]
        positions = sql_bind_positions(sql, dialect=dialect)
        parts, offset = [], 0
        for item in positions:
            parts.extend((sql[offset:item.start], ":plan_bind_" + item.name))
            offset = item.end
        parts.append(sql[offset:])
        named_sql = "".join(parts)
        parser = Dialect.get_or_raise(dialect).parser(error_level=ErrorLevel.RAISE, error_message_context=0)
        tree = parser.parse(Dialect.get_or_raise(dialect).tokenize(named_sql), named_sql)[0]
        for node in tree.find_all(exp.Func):
            if isinstance(node, (exp.Case, exp.Connector, exp.If)):
                continue
            name = node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name().upper()
            if name not in _SAFE_FUNCTIONS or isinstance(node.parent, exp.Dot):
                raise SqlPlanReplayRejected("PLAN_FUNCTION_NOT_ALLOWED")
        if any(dtype.this == exp.DataType.Type.USERDEFINED for dtype in tree.find_all(exp.DataType)):
            raise SqlPlanReplayRejected("PLAN_NATIVE_TYPE_UNSUPPORTED")
        approved = set(allowed_objects)
        for scope in traverse_scope(tree):
            for _, source in scope.selected_sources.values():
                if not isinstance(source, exp.Table):
                    continue
                if not isinstance(source.this, exp.Identifier) or source.catalog or not source.db:
                    raise SqlPlanReplayRejected("PLAN_OBJECT_SCOPE_UNRESOLVED")
                reference = source.db + "." + source.name
                # Oracle未引用的名称由库折叠为大写；PG/MySQL不跨大小写猜测对象等价。
                if database_type == "ORACLE":
                    reference = ".".join(part.name if part.args.get("quoted") else part.name.upper() for part in (source.args["db"], source.this))
                    approved = {name.upper() for name in allowed_objects}
                elif database_type == "POSTGRESQL":
                    reference = ".".join(part.name if part.args.get("quoted") else part.name.lower()
                        for part in (source.args["db"], source.this))
                if reference not in approved:
                    raise SqlPlanReplayRejected("PLAN_OBJECT_SCOPE_DENIED")
        native_types, values = [], []
        for index, parameter in enumerate(parameters, start=1):
            if parameter["position"] != index:
                raise SqlPlanReplayRejected("PLAN_PARAMETER_LAYOUT_INVALID")
            name = _native_type(parameter["database_type_name"], database_type)
            values.append(_decode(parameter["value"], name, database_type))
            native_types.append(name)
        # PostgreSQL保留原生$n并按位置传参；另外两库只做词法绑定替换，不插入原值。
        return SqlPlanReplay(sql=sql if database_type == "POSTGRESQL" else named_sql,
            parameters=tuple(values), native_types=tuple(native_types))
    except SqlPlanReplayRejected:
        raise
    except (ValueError, TypeError, KeyError, InvalidOperation, RecursionError, OverflowError, SqlglotError):
        raise SqlPlanReplayRejected("PLAN_PARAMETER_OR_SOURCE_INVALID") from None
