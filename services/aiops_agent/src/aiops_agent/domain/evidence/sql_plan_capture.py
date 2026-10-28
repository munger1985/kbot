"""受保护材料的静态采集预检；不连接数据库、不授权、不生成可执行语句。"""

from dataclasses import dataclass

from sqlglot import Dialect, exp
from sqlglot.errors import ErrorLevel, SqlglotError
from sqlglot.tokens import TokenType

from aiops_agent.domain.evidence.sql_bindings import sql_bind_positions, SqlBindingInvalid


@dataclass(frozen=True, slots=True)
class SqlPlanCapturePreflightResult:
    status: str
    reason_codes: tuple[str, ...]
    expected_parameter_count: int | None
    statement_shape: str
    required_checks: tuple[str, ...]
    oracle_parameter_peeking: str
    schema_version: str = "SQL_PLAN_CAPTURE_PREFLIGHT.v1"
    execution_authorized: bool = False


_DIALECTS = {"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}
_TEXT_FIELDS = {
    "POSTGRESQL": ("normalized_statement", "sql_text"),
    "MYSQL": ("digest_text", "normalized_statement", "sql_text"),
    "ORACLE": ("sql_text", "normalized_statement"),
}
_FORBIDDEN_NODES = tuple(getattr(exp, name) for name in (
    "Insert", "Update", "Delete", "Merge", "Copy", "Command", "Into", "Lock", "Execute", "Transaction",
) if hasattr(exp, name))


def source_sql_text(fields: dict, database_type: str) -> str | None:
    """只取所选原始统计行中的原生文本；不从其他行、模型或连接环境补齐。"""
    for name in _TEXT_FIELDS.get(database_type, ()):
        value = fields.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def inspect_plan_capture(*, database_type: str, sql: str | None, parameter_count: int) -> SqlPlanCapturePreflightResult:
    """预检通过只表示可进入后续原生核验，绝不表示允许 EXPLAIN 或参数具有代表性。"""
    def result(code=None, *, count=None, shape="UNKNOWN"):
        return SqlPlanCapturePreflightResult(
            status="BLOCKED" if code else "AWAITING_DATABASE_VALIDATION",
            reason_codes=(code,) if code else (), expected_parameter_count=count, statement_shape=shape,
            required_checks=("SOURCE_SQL_REPLAYABILITY", "ORIGINAL_PARSE_CONTEXT", "NATIVE_PARAMETER_TYPES",
                "OBJECT_AND_FUNCTION_ACCESS", "CAPTURE_POLICY", "INDEPENDENT_APPROVAL"),
            oracle_parameter_peeking="NOT_PROVEN" if database_type == "ORACLE" else "NOT_APPLICABLE",
        )
    if database_type not in _DIALECTS:
        return result("DATABASE_TYPE_UNSUPPORTED")
    if not isinstance(sql, str) or not sql.strip():
        return result("SOURCE_SQL_TEXT_MISSING")
    if len(sql) > 20000:
        return result("SOURCE_SQL_LENGTH_EXCEEDED")
    dialect = _DIALECTS[database_type]
    try:
        positions = sql_bind_positions(sql, dialect=dialect)
        # Oracle 数字绑定不被通用 AST 直接支持，只在解析副本中按词法区间替换。
        pieces, offset = [], 0
        for index, item in enumerate(positions):
            pieces.extend((sql[offset:item.start], f":aiops_bind_{index + 1}"))
            offset = item.end
        pieces.append(sql[offset:])
        parsing_sql = "".join(pieces)
        tokens = Dialect.get_or_raise(dialect).tokenize(parsing_sql)
        if len(tokens) > 4096:
            return result("SOURCE_SQL_COMPLEXITY_EXCEEDED")
        depth = 0
        for token in tokens:
            if token.token_type == TokenType.L_PAREN:
                depth += 1
            elif token.token_type == TokenType.R_PAREN:
                depth -= 1
            if depth > 64:
                return result("SOURCE_SQL_COMPLEXITY_EXCEEDED")
        parser = Dialect.get_or_raise(dialect).parser(error_level=ErrorLevel.RAISE, error_message_context=0)
        # 上下文长度为0：第三方解析器的回退警告也不能输出原文或字面量。
        statements = parser.parse(tokens, parsing_sql)
        if len(statements) != 1 or statements[0] is None:
            return result("SINGLE_STATEMENT_REQUIRED")
        tree = statements[0]
        if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return result("ESTIMATE_SELECT_ONLY", shape="OTHER")
        for count, node in enumerate(tree.walk(), start=1):
            if count > 2048 or node.depth > 64:
                return result("SOURCE_SQL_COMPLEXITY_EXCEEDED", shape="SELECT")
            if isinstance(node, _FORBIDDEN_NODES):
                return result("SIDE_EFFECT_OR_LOCK_FORBIDDEN", shape="SELECT")
        placeholders = tuple(tree.find_all(exp.Placeholder))
        if (len(placeholders) != len(positions)
            or {str(item.this or "") for item in placeholders} != {f"aiops_bind_{index + 1}" for index in range(len(positions))}
            or tree.find(exp.Parameter) is not None):
            return result("NATIVE_BIND_STYLE_REQUIRED", shape="SELECT")
        # 无法解释的绑定不能通过未识别即零参数的方式静默放行。
        if database_type == "MYSQL" and any(not item.positional for item in positions):
            return result("NATIVE_BIND_STYLE_REQUIRED", shape="SELECT")
        if database_type in {"ORACLE", "POSTGRESQL"} and any(not item.positional for item in positions):
            return result("NAMED_BIND_MAPPING_REQUIRED", shape="SELECT")
        native_positions = {int(item.name) for item in positions}
        maximum = max(native_positions, default=0)
        if maximum > 64 or (positions and native_positions != set(range(1, maximum + 1))):
            return result("PARAMETER_LAYOUT_INCOMPLETE", shape="SELECT")
        if parameter_count != maximum:
            return result("PARAMETER_COUNT_MISMATCH", count=maximum, shape="SELECT")
        return result(count=maximum, shape="SELECT")
    except (SqlglotError, SqlBindingInvalid, ValueError, TypeError, RecursionError, OverflowError):
        return result("SOURCE_SQL_NOT_REPLAYABLE")
