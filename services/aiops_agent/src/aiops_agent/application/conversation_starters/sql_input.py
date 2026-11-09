"""功能入口按引擎校验 SQL 主体；复用选行与自然语言诊断的身份合同。"""

from aiops_agent.domain.sql_subjects import normalize_sql_subject

_IDENTITIES = {
    "ORACLE": ("ORACLE_SQL_ID", "sql_id"),
    "POSTGRESQL": ("POSTGRESQL_QUERY_ID", "query_id"),
    "MYSQL": ("MYSQL_DIGEST", "digest"),
}


def sql_starter_engine(parameters: dict, database_type: str = "") -> str:
    """无目标的自然语言冻结必须显式给出唯一一种 SQL 身份。"""
    engines = [engine for engine, (_, key) in _IDENTITIES.items() if parameters.get(key) not in (None, "")]
    if len(engines) != 1 or database_type and engines[0] != database_type:
        raise ValueError("请提供与所选数据库匹配的唯一 SQL 标识：Oracle SQL_ID、PostgreSQL Query ID 或 MySQL Digest")
    return engines[0]


def sql_starter_subject(parameters: dict, database_type: str) -> dict:
    engine = sql_starter_engine(parameters, database_type)
    raw = {"subject_type": _IDENTITIES[engine][0], **parameters}
    if "top_level" in raw:
        raw["top_level"] = {"true": True, "false": False}.get(raw["top_level"], raw["top_level"])
    subject = normalize_sql_subject(raw, engine)
    if subject is None or set(parameters) - set(subject):
        raise ValueError("SQL 标识或统计主体范围不正确，请核对数据库类型、OID 和 Schema")
    return subject
