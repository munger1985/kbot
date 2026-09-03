"""诊断 SQL 和类型化参数的离线安全校验。"""

from __future__ import annotations

import re
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from .contracts import DiagnosticParameter, DiagnosticToolDefinition


_BIND_PATTERN = re.compile(r":([a-z][a-z0-9_]*)\b", re.IGNORECASE)
_DANGEROUS = re.compile(
    r"\b("
    r"insert|update|delete|merge|replace|create|alter|drop|truncate|"
    r"grant|revoke|commit|rollback|savepoint|call|execute|exec|begin|"
    r"declare|lock|for\s+update|into\s+(?:out|dump)file|load_file|"
    r"sleep|benchmark"
    r")\b",
    re.IGNORECASE,
)
_PACKAGE_CALL = re.compile(
    r"\b((?:dbms|utl)_[a-z0-9_$#]+)\s*\.",
    re.IGNORECASE,
)


def _code_tokens(sql: str) -> str:
    """移除注释和字符串，同时保留 SQL 结构供安全扫描。"""
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(sql):
        current = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "code":
            if current == "'" :
                state = "single"
                output.append(" ")
            elif current == '"':
                state = "double"
                output.append(" ")
            elif current == "-" and following == "-":
                state = "line_comment"
                output.extend((" ", " "))
                index += 1
            elif current == "/" and following == "*":
                state = "block_comment"
                output.extend((" ", " "))
                index += 1
            else:
                output.append(current)
        elif state == "single":
            output.append(" ")
            if current == "'" and following == "'":
                output.append(" ")
                index += 1
            elif current == "'":
                state = "code"
        elif state == "double":
            output.append(" ")
            if current == '"' and following == '"':
                output.append(" ")
                index += 1
            elif current == '"':
                state = "code"
        elif state == "line_comment":
            output.append("\n" if current in "\r\n" else " ")
            if current in "\r\n":
                state = "code"
        else:
            output.append(" ")
            if current == "*" and following == "/":
                output.append(" ")
                index += 1
                state = "code"
        index += 1
    if state in {"single", "double", "block_comment"}:
        raise ValueError("SQL 包含未闭合字符串或注释")
    return "".join(output)


def validate_readonly_template(
    sql: str, definition: DiagnosticToolDefinition
) -> None:
    if not sql.strip() or len(sql) > 100_000:
        raise ValueError("SQL 模板为空或过长")
    code = _code_tokens(sql).strip()
    if ";" in code:
        raise ValueError("SQL 模板必须是单条语句且不能包含分号")
    if not re.match(r"^(select|with)\b", code, re.IGNORECASE):
        raise ValueError("诊断模板只能使用 SELECT/WITH")
    dangerous = _DANGEROUS.search(code)
    if dangerous:
        raise ValueError(f"诊断模板包含禁止结构：{dangerous.group(0)}")
    referenced_packages = {
        match.group(1).upper() for match in _PACKAGE_CALL.finditer(code)
    }
    undeclared_packages = referenced_packages - set(
        definition.allowed_packages
    )
    if undeclared_packages:
        raise ValueError(
            "诊断模板包含未声明数据库包："
            + "、".join(sorted(undeclared_packages))
        )
    binds = set(_BIND_PATTERN.findall(code.lower()))
    declared = {item.name for item in definition.parameters}
    if binds != declared:
        raise ValueError(
            f"SQL bind 与参数定义不一致：bind={sorted(binds)} "
            f"parameters={sorted(declared)}"
        )
    if re.search(r"[%][(][a-zA-Z_]", code) or "{{" in code or "${" in code:
        raise ValueError("SQL 模板禁止动态字符串插值")
    validate_readonly_ast(
        sql,
        db_type=str(definition.db_type),
        allowed_packages=set(definition.allowed_packages),
    )


def validate_readonly_ast(
    sql: str,
    *,
    db_type: str,
    allowed_packages: set[str] | None = None,
) -> None:
    """使用显式数据库方言复核固定目录 SQL 的只读结构。"""
    dialect = {
        "ORACLE": "oracle",
        "MYSQL": "mysql",
        "POSTGRESQL": "postgres",
    }.get(db_type)
    if dialect is None:
        raise ValueError("数据库 SQL 方言不受支持")
    try:
        statements = parse(sql, read=dialect)
    except ParseError as exc:
        raise ValueError("SQL 模板无法按声明方言解析") from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("SQL 模板 AST 必须是单条 SELECT")
    expression = statements[0]
    if expression.find(exp.Lock) is not None:
        raise ValueError("SQL 模板禁止锁语义")
    package_allowlist = allowed_packages or set()
    for dot in expression.find_all(exp.Dot):
        if not (
            isinstance(dot.this, exp.Identifier)
            and str(dot.this.name).upper() in package_allowlist
            and isinstance(dot.expression, exp.Func)
        ):
            raise ValueError("SQL 模板禁止未声明包函数或不透明点表达式")
    if any("@" in str(table.name or "") for table in expression.find_all(exp.Table)):
        raise ValueError("SQL 模板禁止数据库链路")


def validate_parameters(
    definition: DiagnosticToolDefinition, values: dict[str, Any]
) -> dict[str, Any]:
    definitions = {item.name: item for item in definition.parameters}
    unknown = set(values) - set(definitions)
    if unknown:
        raise ValueError(f"存在未知诊断参数：{sorted(unknown)}")
    normalized: dict[str, Any] = {}
    for name, parameter in definitions.items():
        if name not in values:
            if parameter.required:
                raise ValueError(f"缺少诊断参数：{name}")
            value = parameter.default
        else:
            value = values[name]
        normalized[name] = _validate_parameter(parameter, value)
    return normalized


def _validate_parameter(parameter: DiagnosticParameter, value: Any) -> Any:
    if parameter.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"参数 {parameter.name} 必须是整数")
        if parameter.minimum is not None and value < parameter.minimum:
            raise ValueError(f"参数 {parameter.name} 小于最小值")
        if parameter.maximum is not None and value > parameter.maximum:
            raise ValueError(f"参数 {parameter.name} 大于最大值")
    elif parameter.type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"参数 {parameter.name} 必须是布尔值")
    else:
        if not isinstance(value, str):
            raise ValueError(f"参数 {parameter.name} 必须是字符串")
        if parameter.max_length is not None and len(value) > parameter.max_length:
            raise ValueError(f"参数 {parameter.name} 超过长度限制")
        if parameter.enum and value not in parameter.enum:
            raise ValueError(f"参数 {parameter.name} 不在允许枚举中")
    return value
