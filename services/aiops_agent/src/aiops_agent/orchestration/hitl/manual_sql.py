"""从评审目录生成可复制 SQL，并校验模型候选 SQL。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from aiops_agent.contracts.hitl import ManualDiagnosticQuery
from aiops_agent.diagnostics import DiagnosticRegistry


_BIND = re.compile(r":([a-z][a-z0-9_]*)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"(;|--|/\*|\*/|@\w|\b(?:insert|update|delete|merge|create|alter|"
    r"drop|truncate|grant|revoke|commit|rollback|begin|declare|call|"
    r"execute|exec|lock|for\s+update|into\s+(?:out|dump)file|"
    r"dbms_|utl_|sleep|benchmark|load_file)\b)",
    re.IGNORECASE,
)
_ORACLE_OBJECT = re.compile(
    r"\b(?:from|join)\s+([a-z0-9_$#.]+)", re.IGNORECASE
)
_ORACLE_ALLOW = re.compile(
    r"^(?:gv\$|v\$|dba_|cdb_|user_|all_|dual$)", re.IGNORECASE
)
_MYSQL_ALLOW = re.compile(
    r"^(?:performance_schema|information_schema|sys)\.", re.IGNORECASE
)


def validate_model_manual_sql(sql: str, *, db_type: str) -> None:
    normalized = sql.strip()
    if not 1 <= len(normalized) <= 20000:
        raise ValueError("人工 SQL 长度无效")
    if not re.match(r"^(select|with)\b", normalized, re.IGNORECASE):
        raise ValueError("人工 SQL 只能是 SELECT/WITH")
    if _FORBIDDEN.search(normalized):
        raise ValueError("人工 SQL 包含禁止结构")
    objects = _ORACLE_OBJECT.findall(normalized)
    if not objects and not re.match(
        r"^select\b", normalized, re.IGNORECASE
    ):
        raise ValueError("人工 SQL 没有可验证的数据对象")
    for raw in objects:
        name = raw.lower()
        if db_type == "ORACLE":
            leaf = name.split(".")[-1]
            if not _ORACLE_ALLOW.match(leaf):
                raise ValueError("Oracle 人工 SQL 引用了未授权对象")
        elif not _MYSQL_ALLOW.match(name):
            raise ValueError("MySQL 人工 SQL 引用了未授权对象")
    if len(re.findall(r"\bjoin\b", normalized, re.IGNORECASE)) > 4:
        raise ValueError("人工 SQL Join 数量超限")


class ManualSqlBuilder:
    def __init__(self, registry: DiagnosticRegistry):
        self._registry = registry

    def from_catalog(
        self,
        *,
        tool_snapshot: dict[str, Any],
        db_type: str,
        parameters: dict[str, Any],
        query_id: str,
        purpose: str,
        diagnostic_question: str,
        supports_if: str,
        contradicts_if: str,
    ) -> ManualDiagnosticQuery:
        tool = self._registry.resolve_exact(
            tool_id=tool_snapshot["tool_id"],
            tool_version=tool_snapshot["version"],
            db_type=db_type,
            variant=tool_snapshot["variant"],
            template_sha256=tool_snapshot["template_sha256"],
        )
        values = self._registry.validate_parameters(tool, parameters)
        sql = _BIND.sub(
            lambda match: self._literal(values[match.group(1)]),
            tool.sql,
        )
        validate_model_manual_sql(sql, db_type=db_type)
        definition = tool.definition
        return ManualDiagnosticQuery(
            query_id=query_id,
            origin="CATALOG",
            purpose=purpose,
            diagnostic_question=diagnostic_question,
            sql_text=sql,
            sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
            expected_columns=tuple(
                item.name for item in definition.output_columns
            ),
            expected_types=tuple(
                item.logical_type for item in definition.output_columns
            ),
            expected_shape=(
                "SINGLE_ROW"
                if definition.max_rows == 1
                else "ROW_SET"
            ),
            max_rows=definition.max_rows,
            timeout_hint_seconds=definition.timeout_seconds,
            cost_warning=f"目录评估开销：{definition.cost_level}",
            sensitivity_labels=tuple(
                item.name
                for item in definition.output_columns
                if item.sensitivity != "PUBLIC"
            ),
            supports_if=supports_if,
            contradicts_if=contradicts_if,
        )

    @staticmethod
    def _literal(value: Any) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            if len(value) > 256:
                raise ValueError("人工 SQL 字符串参数过长")
            return "'" + value.replace("'", "''") + "'"
        raise ValueError("人工 SQL 参数类型不受支持")
