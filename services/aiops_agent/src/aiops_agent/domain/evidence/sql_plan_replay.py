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
    relations: tuple[tuple[str, str], ...] = field(repr=False)
    functions: tuple[str, ...]
    operators: tuple[str, ...]


# 只列已实现的内置函数族；不把用户函数或包函数因名字近似而当成纯函数。
_SAFE_FUNCTIONS = frozenset({
    "ABS", "AVG", "CAST", "CEIL", "COALESCE", "COUNT", "DATE", "DATE_TRUNC", "TIMESTAMP_TRUNC",
    "EXTRACT", "FLOOR", "GREATEST", "LEAST", "LENGTH", "LOWER", "MAX", "MIN", "NULLIF", "ROUND",
    "SUM", "TRIM", "UPPER", "SUBSTRING", "STR_POSITION", "STDDEV", "VARIANCE",
})
# 语法树的标准函数名不一定是原生名字；例如PG的date_trunc解析为TIMESTAMP_TRUNC。
# 只接受各引擎确实提供的拼写，随后仍须在原连接核对实际目录及覆盖风险。
_NATIVE_FUNCTIONS = {
    "POSTGRESQL": frozenset({"abs", "avg", "cast", "ceil", "ceiling", "coalesce", "count", "date",
        "date_trunc", "extract", "floor", "greatest", "least", "length", "char_length", "character_length",
        "lower", "max", "min", "nullif", "round", "sum", "trim", "btrim", "upper", "substring", "substr",
        "position", "strpos", "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp"}),
    "MYSQL": frozenset({"abs", "avg", "cast", "ceil", "ceiling", "coalesce", "count", "date", "extract",
        "floor", "greatest", "least", "length", "char_length", "character_length", "lower", "max", "min",
        "nullif", "round", "sum", "trim", "upper", "substring", "substr", "position", "locate",
        "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp"}),
    "ORACLE": frozenset({"abs", "avg", "cast", "ceil", "coalesce", "count", "extract", "floor",
        "greatest", "least", "length", "lower", "max", "min", "nullif", "round", "sum", "trim", "upper",
        "substr", "instr", "stddev", "variance"}),
}
_OPERATORS = {"eq": "=", "neq": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=",
    "add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%", "dpipe": "||",
    "like": "~~", "ilike": "~~*", "bitwiseand": "&", "bitwiseor": "|", "bitwisexor": "#"}
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
        functions = set()
        operators = {"=", "<", ">"}  # 排序、分组和索引路径也可能使用比较操作符。
        # 通用解析器可能消去一元+，不能据此漏掉原生操作符的实际求值入口。
        for token in Dialect.get_or_raise(dialect).tokenize(named_sql):
            spelling = named_sql[token.start:token.end + 1]
            if re.fullmatch(r"[+*/<>=~!@#%^&|`?\-]+", spelling):
                operators.add(spelling)
        if any(isinstance(node, exp.Neg) for node in tree.walk()):
            operators.add("-")
        for node in tree.find_all(exp.Binary):
            if isinstance(node, (exp.And, exp.Or, exp.Is)):
                continue
            if node.key not in _OPERATORS:
                raise SqlPlanReplayRejected("PLAN_OPERATOR_NOT_ALLOWED")
            operators.add(_OPERATORS[node.key])
            if node.key in {"like", "ilike"}:
                operators.add("!" + _OPERATORS[node.key])
        for node in tree.find_all(exp.Func):
            if isinstance(node, (exp.Case, exp.Connector, exp.If)):
                continue
            name = node.name.upper() if isinstance(node, exp.Anonymous) else node.sql_name().upper()
            if name not in _SAFE_FUNCTIONS or isinstance(node.parent, exp.Dot):
                raise SqlPlanReplayRejected("PLAN_FUNCTION_NOT_ALLOWED")
            start, end = node.meta.get("start"), node.meta.get("end")
            spelling = (named_sql[start:end + 1] if isinstance(start, int) and isinstance(end, int)
                else node.sql(dialect=dialect).split("(", 1)[0]).strip().lower()
            if spelling not in _NATIVE_FUNCTIONS[database_type]:
                raise SqlPlanReplayRejected("PLAN_FUNCTION_NOT_ALLOWED")
            functions.add(spelling)
        if any(isinstance(node, exp.Collate) for node in tree.walk()):
            raise SqlPlanReplayRejected("PLAN_COLLATION_NOT_ALLOWED")
        if any(isinstance(node, (exp.TableSample, exp.Hint)) for node in tree.walk()):
            raise SqlPlanReplayRejected("PLAN_PARSE_EXTENSION_NOT_ALLOWED")
        if any(dtype.this == exp.DataType.Type.USERDEFINED for dtype in tree.find_all(exp.DataType)):
            raise SqlPlanReplayRejected("PLAN_NATIVE_TYPE_UNSUPPORTED")
        approved = set(allowed_objects)
        relations = set()
        for scope in traverse_scope(tree):
            visible, parent = set(scope.selected_sources), scope.parent
            while parent is not None:
                visible.update(parent.selected_sources)
                parent = parent.parent
            # 库名/包名/序列伪列不能借Column节点绕过函数和基础对象检查。
            if any(column.table and column.table not in visible for column in scope.columns):
                raise SqlPlanReplayRejected("PLAN_OBJECT_SCOPE_UNRESOLVED")
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
                parts = (source.args["db"], source.this)
                if database_type == "POSTGRESQL":
                    relations.add(tuple(part.name if part.args.get("quoted") else part.name.lower() for part in parts))
                elif database_type == "ORACLE":
                    relations.add(tuple(part.name if part.args.get("quoted") else part.name.upper() for part in parts))
                else:
                    relations.add(tuple(part.name for part in parts))
        if not relations or len(relations) > 32:
            raise SqlPlanReplayRejected("PLAN_CATALOG_OBJECT_LIMIT")
        native_types, values = [], []
        for index, parameter in enumerate(parameters, start=1):
            if parameter["position"] != index:
                raise SqlPlanReplayRejected("PLAN_PARAMETER_LAYOUT_INVALID")
            name = _native_type(parameter["database_type_name"], database_type)
            values.append(_decode(parameter["value"], name, database_type))
            native_types.append(name)
        # PostgreSQL保留原生$n并按位置传参；另外两库只做词法绑定替换，不插入原值。
        return SqlPlanReplay(sql=sql if database_type == "POSTGRESQL" else named_sql,
            parameters=tuple(values), native_types=tuple(native_types), relations=tuple(sorted(relations)),
            functions=tuple(sorted(functions)), operators=tuple(sorted(operators)))
    except SqlPlanReplayRejected:
        raise
    except (ValueError, TypeError, KeyError, InvalidOperation, RecursionError, OverflowError, SqlglotError):
        raise SqlPlanReplayRejected("PLAN_PARAMETER_OR_SOURCE_INVALID") from None
