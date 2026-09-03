"""Oracle 动态只读查询的 AST 策略与确定性归一化。"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp, parse
from sqlglot.errors import ParseError


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DIAGNOSTIC_OBJECT = re.compile(
    r"^(?:"
    r"v\$[a-z0-9_$#]+|gv\$[a-z0-9_$#]+|"
    r"v_\$[a-z0-9_$#]+|gv_\$[a-z0-9_$#]+|"
    r"dba_[a-z0-9_$#]+|cdb_[a-z0-9_$#]+|"
    r"all_[a-z0-9_$#]+|dual"
    r")$",
    re.IGNORECASE,
)
_SAFE_FUNCTIONS = frozenset(
    {
        "ABS",
        "AVG",
        "CAST",
        "CEIL",
        "COALESCE",
        "COUNT",
        "CURRENT_DATE",
        "CURRENT_TIMESTAMP",
        "DECODE",
        "DENSE_RANK",
        "FLOOR",
        "GREATEST",
        "LAG",
        "LEAD",
        "LEAST",
        "LENGTH",
        "LISTAGG",
        "LOWER",
        "MAX",
        "MIN",
        "NVL",
        "NVL2",
        "PERCENT_RANK",
        "RANK",
        "REGEXP_REPLACE",
        "REPLACE",
        "ROUND",
        "ROW_NUMBER",
        "RTRIM",
        "STDDEV",
        "SUBSTR",
        "SUM",
        "TO_CHAR",
        "TO_DATE",
        "TO_NUMBER",
        "TRIM",
        "TRUNC",
        "UPPER",
        "VARIANCE",
    }
)

# sqlglot 的 AST 使用跨方言内部函数名。策略契约和模型提示仍应使用
# Oracle 表面函数名，否则合法的 Oracle SQL 会因解析器内部改名被误拦截。
_SQLGLOT_ORACLE_FUNCTION_NAMES = {
    "STR_TO_DATE": "TO_DATE",
    "SUBSTRING": "SUBSTR",
}


class DynamicQueryRejected(ValueError):
    """动态查询未通过确定性策略。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DynamicQueryPolicySnapshot(BaseModel):
    """规划端和执行端共同消费的不可变查询边界。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "ORACLE_DYNAMIC_QUERY_POLICY.v4"
    allowed_objects: tuple[str, ...] = ()
    allowed_functions: tuple[str, ...] = tuple(sorted(_SAFE_FUNCTIONS))
    allowed_packages: tuple[str, ...] = ("DBMS_XPLAN",)
    max_rows: int = Field(default=200, ge=1, le=1000)
    max_sql_chars: int = Field(default=20_000, ge=1, le=100_000)
    max_bind_count: int = Field(default=32, ge=0, le=128)
    allow_catalog_object_families: bool = True


class ValidatedDynamicQuery(BaseModel):
    """可以写入 Grant 的规范化动态查询，不包含数据库凭据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "ORACLE_VALIDATED_DYNAMIC_QUERY.v3"
    normalized_sql: str
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    referenced_objects: tuple[str, ...]
    projected_columns: tuple[str, ...]
    column_sensitivities: tuple[Literal["PUBLIC", "MASKED", "HASHED"], ...]
    bind_names: tuple[str, ...]
    parameters: dict[str, str | int | float | bool | None]
    max_rows: int
    execution_decision: Literal["AUTO_EXECUTE", "APPROVAL_REQUIRED"]
    approval_reason_codes: tuple[str, ...] = ()


class OracleDynamicQueryPolicy:
    """使用 Oracle AST 验证单条、显式投影、无副作用查询。"""

    def __init__(self, snapshot: DynamicQueryPolicySnapshot) -> None:
        self.snapshot = snapshot
        self._allowed_objects = {
            self._canonical_object(value)
            for value in snapshot.allowed_objects
        }
        self._allowed_functions = {
            value.upper() for value in snapshot.allowed_functions
        }
        self._allowed_packages = {
            value.upper() for value in snapshot.allowed_packages
        }

    def validate(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> ValidatedDynamicQuery:
        if not sql.strip() or len(sql) > self.snapshot.max_sql_chars:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LENGTH_INVALID",
                "动态 SQL 为空或超过长度限制",
            )
        try:
            statements = parse(sql, read="oracle")
        except ParseError as exc:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_PARSE_FAILED",
                "动态 SQL 无法按 Oracle 方言解析",
            ) from exc
        if len(statements) != 1 or statements[0] is None:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_MULTIPLE_STATEMENTS",
                "动态 SQL 必须且只能包含一条语句",
            )
        expression = statements[0]
        if not isinstance(expression, exp.Select):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_NOT_SELECT",
                "动态 SQL 只允许单条 SELECT 或带 WITH 的 SELECT",
            )
        self._validate_nodes(expression)
        projected_columns = self._projected_columns(expression)
        column_sensitivities = ("PUBLIC",) * len(projected_columns)
        referenced_objects = self._referenced_objects(expression)
        bind_names = self._bind_names(expression)
        normalized_parameters = self._parameters(bind_names, parameters or {})
        effective_rows = self._effective_row_limit(expression)
        limited = expression.copy().limit(effective_rows, copy=False)
        normalized_sql = limited.sql(dialect="oracle", comments=False)
        policy_payload = json.dumps(
            self.snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return ValidatedDynamicQuery(
            normalized_sql=normalized_sql,
            query_sha256=self._sha256(normalized_sql),
            policy_sha256=self._sha256(policy_payload),
            referenced_objects=referenced_objects,
            projected_columns=projected_columns,
            column_sensitivities=column_sensitivities,
            bind_names=bind_names,
            parameters=normalized_parameters,
            max_rows=effective_rows,
            execution_decision="AUTO_EXECUTE",
            approval_reason_codes=(),
        )

    def _effective_row_limit(self, expression: exp.Select) -> int:
        limit = expression.args.get("limit")
        if limit is None:
            return self.snapshot.max_rows
        options = limit.args.get("limit_options")
        if options is not None and (
            options.args.get("percent") or options.args.get("with_ties")
        ):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LIMIT_INVALID",
                "动态 SQL 禁止百分比或WITH TIES行数限制",
            )
        value = limit.args.get("count") or limit.args.get("expression")
        if not isinstance(value, exp.Literal) or value.is_string:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LIMIT_INVALID",
                "动态 SQL 行数限制必须是正整数常量",
            )
        try:
            requested = int(value.this)
        except (TypeError, ValueError) as exc:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LIMIT_INVALID",
                "动态 SQL 行数限制必须是正整数常量",
            ) from exc
        if requested <= 0:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LIMIT_INVALID",
                "动态 SQL 行数限制必须大于零",
            )
        return min(requested, self.snapshot.max_rows)

    def _validate_nodes(self, expression: exp.Select) -> None:
        if expression.find(exp.Lock) is not None:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_LOCK_FORBIDDEN",
                "动态 SQL 禁止 FOR UPDATE 或其他锁语义",
            )
        package_functions: set[int] = set()
        for dot in expression.find_all(exp.Dot):
            if not self._allowed_package_call(dot):
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_PACKAGE_CALL_FORBIDDEN",
                    "动态 SQL 只允许策略声明的只读包函数",
                )
            package_functions.add(id(dot.expression))
        table_wrappers = {
            id(table.this)
            for table in expression.find_all(exp.Table)
            if self._table_function_package(table) is not None
        }
        for function in expression.find_all(exp.Func):
            # sqlglot 将 AND/OR 连接符也纳入 Func 继承体系；它们是 SQL
            # 语法节点而非可调用函数，不能参与函数白名单判断。
            if isinstance(function, exp.Connector):
                continue
            if (
                id(function) in package_functions
                or id(function) in table_wrappers
            ):
                continue
            name = function.sql_name().upper()
            if name == "ANONYMOUS":
                name = str(getattr(function, "name", "")).upper()
            name = _SQLGLOT_ORACLE_FUNCTION_NAMES.get(name, name)
            if name not in self._allowed_functions:
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_FUNCTION_FORBIDDEN",
                    f"动态 SQL 函数不在允许清单：{name or 'UNKNOWN'}",
                )

    def _allowed_package_call(self, expression: exp.Dot) -> bool:
        package = expression.this
        return (
            isinstance(package, exp.Identifier)
            and str(package.name).upper() in self._allowed_packages
            and isinstance(expression.expression, exp.Func)
        )

    def _table_function_package(self, table: exp.Table) -> str | None:
        wrapper = table.this
        if not isinstance(wrapper, exp.Anonymous):
            return None
        if str(wrapper.name).upper() != "TABLE":
            return None
        calls = [
            dot
            for dot in wrapper.find_all(exp.Dot)
            if self._allowed_package_call(dot)
        ]
        if len(calls) != 1:
            return None
        return str(calls[0].this.name).upper()

    def _projected_columns(self, expression: exp.Select) -> tuple[str, ...]:
        columns: list[str] = []
        for projection in expression.expressions:
            if projection.is_star:
                columns.append("*")
                continue
            name = str(projection.alias_or_name or "").lower()
            if not name or not _IDENTIFIER.fullmatch(name):
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_COLUMN_ALIAS_REQUIRED",
                    "每个返回表达式都必须具有简单且唯一的列名",
                )
            columns.append(name)
        if not columns or len(columns) != len(set(columns)):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_COLUMNS_INVALID",
                "动态 SQL 返回列不能为空或重名",
            )
        return tuple(columns)

    def _referenced_objects(self, expression: exp.Select) -> tuple[str, ...]:
        cte_names = {
            str(cte.alias_or_name).lower()
            for cte in expression.find_all(exp.CTE)
        }
        objects = {
            f"sys.{str(dot.this.name).lower()}"
            for dot in expression.find_all(exp.Dot)
            if self._allowed_package_call(dot)
        }
        for table in expression.find_all(exp.Table):
            if self._table_function_package(table) is not None:
                continue
            name = str(table.name or "")
            if name.lower() in cte_names:
                continue
            if "@" in name:
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_DATABASE_LINK_FORBIDDEN",
                    "动态 SQL 禁止数据库链路",
                )
            owner = str(table.db or "")
            canonical = self._canonical_object(
                f"{owner}.{name}" if owner else name
            )
            if owner and owner.upper() != "SYS":
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_SCHEMA_FORBIDDEN",
                    f"动态 SQL 禁止访问 Schema：{owner}",
                )
            short_name = name.lower()
            family_allowed = (
                self.snapshot.allow_catalog_object_families
                and _DIAGNOSTIC_OBJECT.fullmatch(short_name) is not None
            )
            if canonical not in self._allowed_objects and not family_allowed:
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_OBJECT_FORBIDDEN",
                    f"动态 SQL 对象不在诊断范围：{canonical}",
                )
            objects.add(canonical)
        if not objects:
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_OBJECT_REQUIRED",
                "动态 SQL 必须读取至少一个受控诊断对象",
            )
        return tuple(sorted(objects))

    def _bind_names(self, expression: exp.Select) -> tuple[str, ...]:
        names = tuple(
            sorted(
                {
                    str(placeholder.this).lower()
                    for placeholder in expression.find_all(exp.Placeholder)
                }
            )
        )
        if len(names) > self.snapshot.max_bind_count or any(
            _IDENTIFIER.fullmatch(name) is None for name in names
        ):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_BINDS_INVALID",
                "动态 SQL bind 名称或数量不符合策略",
            )
        return names

    @staticmethod
    def _parameters(
        bind_names: tuple[str, ...], parameters: dict[str, Any]
    ) -> dict[str, str | int | float | bool | None]:
        normalized = {str(key).lower(): value for key, value in parameters.items()}
        if set(normalized) != set(bind_names):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_PARAMETERS_MISMATCH",
                "动态 SQL bind 与参数不一致",
            )
        for name, value in normalized.items():
            if value is not None and not isinstance(
                value, (str, int, float, bool)
            ):
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_PARAMETER_TYPE_INVALID",
                    f"动态 SQL 参数类型不受支持：{name}",
                )
            if isinstance(value, str) and len(value) > 4000:
                raise DynamicQueryRejected(
                    "DYNAMIC_SQL_PARAMETER_LENGTH_INVALID",
                    f"动态 SQL 参数过长：{name}",
                )
        return normalized

    @staticmethod
    def _canonical_object(value: str) -> str:
        return value.strip().lower()

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
