"""MySQL 长尾只读诊断的独立方言及副作用策略。"""

from __future__ import annotations

from typing import Any

from sqlglot import exp

from .dynamic_query import DynamicQueryRejected, PostgreSQLDynamicQueryPolicy, ValidatedMySQLQuery


class MySQLDynamicQueryPolicy(PostgreSQLDynamicQueryPolicy):
    """复用通用 AST 结构守卫，使用独立 MySQL 方言、对象及函数策略。"""

    dialect = "mysql"
    result_model = ValidatedMySQLQuery

    def _validate_nodes(self, expression: exp.Select) -> None:
        super()._validate_nodes(expression)
        forbidden = ("PropertyEQ", "SessionParameter", "Parameter", "Into", "Lock")
        if any(expression.find(getattr(exp, name)) is not None for name in forbidden if hasattr(exp, name)):
            raise DynamicQueryRejected(
                "DYNAMIC_SQL_SIDE_EFFECT_FORBIDDEN",
                "MySQL 动态诊断禁止变量、赋值、文件输出及锁定",
            )

    def validate(self, sql: str, parameters: dict[str, Any] | None = None, *, explain: bool = False):
        if explain:
            raise DynamicQueryRejected("DYNAMIC_SQL_NOT_SELECT", "计划取证需要单独工具授权")
        return super().validate(sql, parameters, explain=False)
