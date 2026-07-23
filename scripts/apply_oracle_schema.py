"""将 KBot 4.0 规范 DDL 应用到空白 Oracle Schema。"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "database" / "oracle"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_core.database.oracle import create_database_runtime  # noqa: E402


SERVICE_ORDER = (
    "platform_core",
    "model_serving",
    "knowledge_core",
    "agent_runtime",
)


@dataclass(frozen=True)
class SchemaStatement:
    """带来源位置的单条 DDL。"""

    script: Path
    ordinal: int
    sql: str

    @property
    def label(self) -> str:
        """返回适合日志展示且不包含连接凭据的 DDL 摘要。"""
        without_comments = re.sub(
            r"^\s*(?:(?:--[^\n]*(?:\n|$))|(?:/\*.*?\*/\s*))*",
            "",
            self.sql,
            flags=re.DOTALL,
        )
        normalized = re.sub(r"\s+", " ", without_comments).strip()
        match = re.match(
            r"(CREATE(?:\s+OR\s+REPLACE)?\s+(?:TABLE|VIEW|INDEX)"
            r"|ALTER\s+TABLE|COMMENT\s+ON\s+COLUMN)\s+([A-Z0-9_.$]+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            return f"{match.group(1).upper()} {match.group(2).upper()}"
        return normalized[:100]


def split_oracle_statements(sql: str) -> list[str]:
    """按分号拆分普通 Oracle DDL，同时正确处理字符串和注释。"""
    statements: list[str] = []
    current: list[str] = []
    index = 0
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""

        if in_line_comment:
            current.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue

        if in_block_comment:
            current.append(char)
            if char == "*" and next_char == "/":
                current.append(next_char)
                index += 2
                in_block_comment = False
            else:
                index += 1
            continue

        if in_string:
            current.append(char)
            if char == "'" and next_char == "'":
                current.append(next_char)
                index += 2
                continue
            if char == "'":
                in_string = False
            index += 1
            continue

        if char == "-" and next_char == "-":
            current.extend((char, next_char))
            index += 2
            in_line_comment = True
            continue
        if char == "/" and next_char == "*":
            current.extend((char, next_char))
            index += 2
            in_block_comment = True
            continue
        if char == "'":
            current.append(char)
            index += 1
            in_string = True
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    trailing = "".join(current).strip()
    if trailing:
        statements.append(trailing)
    if in_string or in_block_comment:
        raise ValueError("SQL 文件存在未闭合的字符串或块注释")
    return statements


def load_schema_statements() -> list[SchemaStatement]:
    """按服务依赖和文件名前缀读取全部规范 DDL。"""
    result: list[SchemaStatement] = []
    for service in SERVICE_ORDER:
        scripts = sorted((SCHEMA_ROOT / service).glob("[0-9][0-9][0-9]_*.sql"))
        if not scripts:
            raise RuntimeError(f"{service} 没有规范 DDL")
        for script in scripts:
            statements = split_oracle_statements(
                script.read_text(encoding="utf-8")
            )
            for ordinal, statement in enumerate(statements, start=1):
                result.append(
                    SchemaStatement(
                        script=script.relative_to(ROOT),
                        ordinal=ordinal,
                        sql=statement,
                    )
                )
    return result


async def _read_target(connection: AsyncConnection) -> tuple[str, str]:
    row = (
        await connection.execute(
            text(
                """
                SELECT
                    SYS_CONTEXT('USERENV', 'CON_NAME'),
                    SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                FROM dual
                """
            )
        )
    ).one()
    return str(row[0]), str(row[1])


async def _assert_empty_schema(connection: AsyncConnection) -> None:
    existing = (
        await connection.execute(
            text(
                """
                SELECT object_name, object_type
                FROM user_objects
                WHERE object_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND object_type IN ('TABLE', 'VIEW')
                ORDER BY object_type, object_name
                """
            )
        )
    ).all()
    if existing:
        rendered = ", ".join(f"{name}({kind})" for name, kind in existing)
        raise RuntimeError(
            "目标 Schema 已存在 KBot 表或视图，初始化已拒绝："
            f"{rendered}"
        )


async def _assert_ddl_privileges(connection: AsyncConnection) -> None:
    """在执行第一条 DDL 前检查最小权限和默认表空间额度。"""
    privileges = set(
        (
            await connection.execute(
                text(
                    """
                    SELECT privilege
                    FROM session_privs
                    WHERE privilege IN (
                        'CREATE TABLE',
                        'CREATE VIEW',
                        'UNLIMITED TABLESPACE'
                    )
                    """
                )
            )
        ).scalars()
    )
    missing = {"CREATE TABLE", "CREATE VIEW"} - privileges
    if missing:
        raise RuntimeError(
            "当前用户缺少 DDL 权限："
            f"{', '.join(sorted(missing))}；请由 PDB 管理员授权"
        )

    default_tablespace = (
        await connection.execute(
            text("SELECT default_tablespace FROM user_users")
        )
    ).scalar_one()
    if "UNLIMITED TABLESPACE" not in privileges:
        quota = (
            await connection.execute(
                text(
                    """
                    SELECT max_bytes
                    FROM user_ts_quotas
                    WHERE tablespace_name = :tablespace_name
                    """
                ),
                {"tablespace_name": default_tablespace},
            )
        ).scalar_one_or_none()
        if quota is None or quota == 0:
            raise RuntimeError(
                f"当前用户在默认表空间 {default_tablespace} 没有额度；"
                "请由 PDB 管理员设置应用表空间和 QUOTA"
            )

    segment_management = (
        await connection.execute(
            text(
                """
                SELECT segment_space_management
                FROM user_tablespaces
                WHERE tablespace_name = :tablespace_name
                """
            ),
            {"tablespace_name": default_tablespace},
        )
    ).scalar_one_or_none()
    if segment_management != "AUTO":
        raise RuntimeError(
            f"默认表空间 {default_tablespace} 不是 ASSM；"
            "Oracle VECTOR 要求 SEGMENT SPACE MANAGEMENT AUTO"
        )


async def _validate_schema(
    connection: AsyncConnection,
    expected_tables: set[str],
    expected_views: set[str],
) -> None:
    rows = (
        await connection.execute(
            text(
                """
                SELECT object_name, object_type, status
                FROM user_objects
                WHERE object_name LIKE 'KBOT\\_%' ESCAPE '\\'
                  AND object_type IN ('TABLE', 'VIEW')
                ORDER BY object_type, object_name
                """
            )
        )
    ).all()
    actual_tables = {name for name, kind, _ in rows if kind == "TABLE"}
    actual_views = {name for name, kind, _ in rows if kind == "VIEW"}
    invalid = [(name, kind) for name, kind, status in rows if status != "VALID"]
    if actual_tables != expected_tables:
        raise RuntimeError(
            "建表结果不完整："
            f"缺少={sorted(expected_tables - actual_tables)}，"
            f"多出={sorted(actual_tables - expected_tables)}"
        )
    if actual_views != expected_views:
        raise RuntimeError(
            "视图结果不完整："
            f"缺少={sorted(expected_views - actual_views)}，"
            f"多出={sorted(actual_views - expected_views)}"
        )
    if invalid:
        raise RuntimeError(f"存在无效对象：{invalid}")


async def apply_schema(dry_run: bool) -> None:
    """执行空库检查、DDL 和对象完整性校验。"""
    statements = load_schema_statements()
    expected_tables = {
        match.group(1).upper()
        for item in statements
        if (
            match := re.match(
                r"\s*(?:--[^\n]*\n\s*)*CREATE\s+TABLE\s+([A-Z0-9_]+)",
                item.sql,
                flags=re.IGNORECASE,
            )
        )
    }
    expected_views = {
        match.group(1).upper()
        for item in statements
        if (
            match := re.match(
                r"\s*(?:--[^\n]*\n\s*)*CREATE\s+OR\s+REPLACE\s+VIEW"
                r"\s+([A-Z0-9_]+)",
                item.sql,
                flags=re.IGNORECASE,
            )
        )
    }
    if dry_run:
        print(
            f"DDL 解析通过：{len(statements)} 条语句，"
            f"{len(expected_tables)} 张表，{len(expected_views)} 个视图"
        )
        return

    runtime = create_database_runtime()
    try:
        async with runtime.engine.connect() as connection:
            pdb_name, schema_name = await _read_target(connection)
            await _assert_empty_schema(connection)
            await _assert_ddl_privileges(connection)
            print(f"目标确认：PDB={pdb_name}，Schema={schema_name}")

            for position, statement in enumerate(statements, start=1):
                try:
                    await connection.exec_driver_sql(statement.sql)
                except Exception:
                    print(
                        "DDL 执行失败："
                        f"{statement.script} 第 {statement.ordinal} 条，"
                        f"{statement.label}"
                    )
                    raise
                print(
                    f"[{position}/{len(statements)}] "
                    f"{statement.script}: {statement.label}"
                )

            await _validate_schema(
                connection,
                expected_tables=expected_tables,
                expected_views=expected_views,
            )
            print(
                f"Schema 初始化完成：{len(expected_tables)} 张表，"
                f"{len(expected_views)} 个视图"
            )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 KBot 4.0 规范 DDL 应用到空白 Oracle Schema"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析并统计 DDL，不连接或修改数据库",
    )
    args = parser.parse_args()
    try:
        asyncio.run(apply_schema(dry_run=args.dry_run))
    except RuntimeError as exc:
        print(f"Schema 初始化拒绝：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
