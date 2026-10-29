"""仅从历史原始证据提取 SQL 文本；不重放语句或生成计划。"""

_TEXT_FIELDS = {
    "POSTGRESQL": ("normalized_statement", "sql_text"),
    "MYSQL": ("digest_text", "normalized_statement", "sql_text"),
    "ORACLE": ("sql_text", "normalized_statement"),
}


def source_sql_text(fields: dict, database_type: str) -> str | None:
    """只取所选原始统计行中的原生文本；不从其他行、模型或连接环境补齐。"""
    for name in _TEXT_FIELDS.get(database_type, ()):
        value = fields.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None
