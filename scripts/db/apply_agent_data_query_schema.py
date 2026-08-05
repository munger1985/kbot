"""为现有 Oracle Schema 安全补齐 Agent 问数模式字段与约束。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sqlalchemy import text

from platform_core.database.oracle import create_database_runtime


TABLE_NAME = "KBOT_AGENT_DEFINITION"
CONSTRAINT_NAME = "CK_AGENT_DEF_DQ_MODE"


@dataclass(frozen=True)
class ColumnMetadata:
    """现有 Oracle 字段的必要结构信息。"""

    data_type: str
    char_length: int | None
    nullable: bool


def build_alignment_statements(
    *,
    columns: dict[str, ColumnMetadata],
    constraint_exists: bool,
    constraint_enabled: bool,
) -> list[str]:
    """按当前元数据生成可重复执行的最小 DDL。"""
    expected = {
        "DATA_QUERY_MODE": 16,
        "DATA_PROFILE_NAME": 256,
    }
    for name, length in expected.items():
        current = columns.get(name)
        if current is None:
            continue
        if (
            current.data_type != "VARCHAR2"
            or current.char_length != length
            or not current.nullable
        ):
            raise RuntimeError(
                f"{TABLE_NAME}.{name} 结构与 Canonical DDL 不一致"
            )

    statements: list[str] = []
    if "DATA_QUERY_MODE" not in columns:
        statements.append(
            f"ALTER TABLE {TABLE_NAME} ADD "
            "(DATA_QUERY_MODE VARCHAR2(16 CHAR))"
        )
    if "DATA_PROFILE_NAME" not in columns:
        statements.append(
            f"ALTER TABLE {TABLE_NAME} ADD "
            "(DATA_PROFILE_NAME VARCHAR2(256 CHAR))"
        )
    if not constraint_exists:
        statements.append(
            f"ALTER TABLE {TABLE_NAME} ADD CONSTRAINT {CONSTRAINT_NAME} "
            "CHECK (DATA_QUERY_MODE IN ('MCP', 'SEMANTIC') "
            "OR DATA_QUERY_MODE IS NULL)"
        )
    elif not constraint_enabled:
        statements.append(
            f"ALTER TABLE {TABLE_NAME} ENABLE VALIDATE "
            f"CONSTRAINT {CONSTRAINT_NAME}"
        )
    statements.extend(
        (
            f"COMMENT ON COLUMN {TABLE_NAME}.DATA_QUERY_MODE IS "
            "'data_query 能力的固定 Provider；只能由 Agent 配置选择 MCP 或 SEMANTIC'",
            f"COMMENT ON COLUMN {TABLE_NAME}.DATA_PROFILE_NAME IS "
            "'SelectAI/AIReport 问数服务使用的 Profile 名称'",
        )
    )
    return statements


async def apply() -> None:
    """对当前连接用户的既有 Agent Runtime Schema 执行幂等对齐。"""
    runtime = create_database_runtime()
    try:
        async with runtime.engine.begin() as connection:
            table_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM USER_TABLES "
                f"WHERE TABLE_NAME='{TABLE_NAME}'"
            ))).scalar_one())
            if table_count != 1:
                raise RuntimeError(
                    f"缺少 {TABLE_NAME}；本脚本只用于对齐既有 4.0 Schema"
                )

            column_rows = (await connection.execute(text(
                "SELECT COLUMN_NAME, DATA_TYPE, CHAR_LENGTH, NULLABLE "
                "FROM USER_TAB_COLUMNS "
                f"WHERE TABLE_NAME='{TABLE_NAME}' AND COLUMN_NAME IN "
                "('DATA_QUERY_MODE','DATA_PROFILE_NAME')"
            ))).all()
            columns = {
                str(row[0]): ColumnMetadata(
                    data_type=str(row[1]),
                    char_length=(int(row[2]) if row[2] is not None else None),
                    nullable=str(row[3]) == "Y",
                )
                for row in column_rows
            }

            constraint_rows = (await connection.execute(text(
                "SELECT TABLE_NAME, STATUS, VALIDATED "
                "FROM USER_CONSTRAINTS "
                f"WHERE CONSTRAINT_NAME='{CONSTRAINT_NAME}'"
            ))).all()
            if constraint_rows and str(constraint_rows[0][0]) != TABLE_NAME:
                raise RuntimeError(
                    f"约束名 {CONSTRAINT_NAME} 已被其他表占用"
                )
            constraint_exists = bool(constraint_rows)
            constraint_enabled = bool(
                constraint_rows
                and str(constraint_rows[0][1]) == "ENABLED"
                and str(constraint_rows[0][2]) == "VALIDATED"
            )

            if "DATA_QUERY_MODE" in columns:
                invalid_count = int((await connection.execute(text(
                    f"SELECT COUNT(*) FROM {TABLE_NAME} "
                    "WHERE DATA_QUERY_MODE IS NOT NULL "
                    "AND DATA_QUERY_MODE NOT IN ('MCP','SEMANTIC')"
                ))).scalar_one())
                if invalid_count:
                    raise RuntimeError(
                        f"发现 {invalid_count} 个无效 DATA_QUERY_MODE，拒绝修改约束"
                    )

            statements = build_alignment_statements(
                columns=columns,
                constraint_exists=constraint_exists,
                constraint_enabled=constraint_enabled,
            )
            for statement in statements:
                await connection.execute(text(statement))

            verification = dict((await connection.execute(text(
                "SELECT COLUMN_NAME, CHAR_LENGTH FROM USER_TAB_COLUMNS "
                f"WHERE TABLE_NAME='{TABLE_NAME}' AND COLUMN_NAME IN "
                "('DATA_QUERY_MODE','DATA_PROFILE_NAME')"
            ))).all())
            if verification != {
                "DATA_QUERY_MODE": 16,
                "DATA_PROFILE_NAME": 256,
            }:
                raise RuntimeError("Agent 问数模式字段核验失败")
            constraint_status = (await connection.execute(text(
                "SELECT STATUS, VALIDATED FROM USER_CONSTRAINTS "
                f"WHERE TABLE_NAME='{TABLE_NAME}' "
                f"AND CONSTRAINT_NAME='{CONSTRAINT_NAME}'"
            ))).one_or_none()
            if constraint_status is None or tuple(constraint_status) != (
                "ENABLED",
                "VALIDATED",
            ):
                raise RuntimeError("Agent 问数模式约束核验失败")
            print("Agent 问数模式 Schema 已对齐并通过核验")
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(apply())
