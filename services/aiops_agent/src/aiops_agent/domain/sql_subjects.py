"""SQL 诊断请求的类型化身份与有界比较范围，不从自然语言猜测主体。"""

from __future__ import annotations

import re


SQL_IDENTITIES = {"POSTGRESQL_QUERY_ID": "query_id", "MYSQL_DIGEST": "digest", "ORACLE_SQL_ID": "sql_id"}
SQL_STATISTICS_TOOLS = {
    "POSTGRESQL": ("db.postgresql.statement.details", "db.postgresql.statement.subjects", "db.postgresql.statement.top", "workload.sql.top_window"),
    "MYSQL": ("db.sql.cursor_details", "db.mysql.statement.digest.top", "workload.sql.top_window"),
    "ORACLE": ("db.sql.cursor_details", "db.sql.top_current", "workload.sql.top_window"),
}
_SQL_ENGINES = {"POSTGRESQL_QUERY_ID": "POSTGRESQL", "MYSQL_DIGEST": "MYSQL", "ORACLE_SQL_ID": "ORACLE"}


def normalize_sql_subject(raw: dict, database_type: str = "") -> dict | None:
    """保留显式 NULL Schema；拒绝跨引擎身份、布尔 OID 和范围溢出。"""
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("subject_type") or "").strip().upper()
    if kind not in SQL_IDENTITIES or database_type not in {"", _SQL_ENGINES[kind]}:
        return None
    identity = SQL_IDENTITIES[kind]
    if any(key in raw for key in SQL_IDENTITIES.values() if key != identity):
        return None
    value = str(raw.get(identity) if raw.get(identity) is not None else "").strip()
    result = {"subject_type": kind}
    if kind == "ORACLE_SQL_ID":
        value = value.lower()
        if len(value) != 13 or not value.isascii() or not value.isalnum():
            return None
        for key in ("dbid", "inst_id", "container_id", "child_number", "plan_hash_value"):
            if key not in raw:
                continue
            entry = raw[key]
            if isinstance(entry, bool) or not re.fullmatch(r"0|[1-9][0-9]{0,19}", str(entry)):
                return None
            number = int(entry)
            if number > 2**64 - 1 or key in {"dbid", "inst_id"} and number == 0:
                return None
            result[key] = number
    elif kind == "MYSQL_DIGEST":
        value = value.upper()
        if not re.fullmatch(r"[0-9A-F]{64}", value):
            return None
        if "schema_name" in raw:
            schema = raw["schema_name"]
            if schema is not None and (not isinstance(schema, str) or not schema or len(schema) > 64):
                return None
            result["schema_name"] = schema
    else:
        if not re.fullmatch(r"-?(?:0|[1-9][0-9]{0,18})", value) or not -(2**63) <= int(value) < 2**63:
            return None
        value = str(int(value))
        for key in ("database_oid", "user_oid"):
            if key not in raw:
                continue
            entry = raw[key]
            if isinstance(entry, bool) or not re.fullmatch(r"[1-9][0-9]{0,9}", str(entry)) or int(entry) > 2**32 - 1:
                return None
            result[key] = int(entry)
        if "top_level" in raw:
            if not isinstance(raw["top_level"], bool):
                return None
            result["top_level"] = raw["top_level"]
    result[identity] = value
    return result


ORACLE_CURSOR_KEYS = ("sql_id", "dbid", "inst_id", "container_id", "child_number", "plan_hash_value")


def normalize_sql_comparison(raw: dict, database_type: str = "") -> tuple[dict, ...] | None:
    """最多四项且请求范围互斥；同一统计行不能被两个比较请求重复计入。"""
    if not isinstance(raw, dict) or set(raw) != {"subjects"}:
        return None
    entries = raw.get("subjects")
    if not isinstance(entries, (list, tuple)) or not 2 <= len(entries) <= 4:
        return None
    subjects = tuple(normalize_sql_subject(item, database_type) for item in entries)
    if any(item is None for item in subjects) or len({item["subject_type"] for item in subjects}) != 1:
        return None
    if any(set(raw_item) - set(subject) for raw_item, subject in zip(entries, subjects, strict=True)):
        return None
    for index, left in enumerate(subjects):
        for right in subjects[index + 1:]:
            if all(left[key] == right[key] for key in left.keys() & right.keys()):
                return None
    return subjects
