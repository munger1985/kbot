"""部署时定向执行通知 v9 新表增量；Oracle DDL 隐式提交，失败后禁止盲目重跑。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQL = ROOT / "database/oracle/operations/upgrade_notification_delivery_9.sql"


def contract(sql: str) -> dict:
    """从已审阅增量 DDL 提取部署验收合同，不触及运行时。"""
    tables = {}
    for name, body in re.findall(r"CREATE TABLE (KBOT_NOTIFY_\w+) \((.*?)\n\);", sql, re.S):
        columns = {}
        for line in body.splitlines():
            match = re.fullmatch(r"\s*(\w+) (RAW\(16\)|VARCHAR2\(\d+ CHAR\)|NUMERIC\(38, 0\)|NUMBER\(19\)|SMALLINT|JSON|TIMESTAMP WITH TIME ZONE)( NOT NULL)?,?", line)
            if match:
                column, kind, required = match.groups()
                columns[column] = [kind, "N" if required else "Y"]
        foreign_keys = sorted([list(item) for item in re.findall(
            r"FOREIGN KEY\((\w+)\) REFERENCES (\w+) \((\w+)\)", body)])
        tables[name] = {"columns": columns, "foreign_keys": foreign_keys}
    indexes = {name: [table, columns.split(", ")] for name, table, columns in re.findall(
        r"CREATE INDEX (IX_NOTIFY_\w+) ON (KBOT_NOTIFY_\w+) \((.*?)\);", sql)}
    if len(tables) != 8 or len(indexes) != 7 or any(not item["columns"] for item in tables.values()):
        raise ValueError("增量 DDL 合同不是预期的八表七索引")
    return {"tables": tables, "indexes": indexes}


async def verify(conn, expected: dict) -> None:
    for table, spec in expected["tables"].items():
        rows = (await conn.execute(text("""
            SELECT column_name, data_type, data_length, char_length,
                   data_precision, data_scale, nullable FROM user_tab_columns
            WHERE table_name=:name
        """), {"name": table})).mappings()
        actual = {}
        for row in rows:
            kind = row["data_type"]
            if kind == "NUMBER":
                precision, scale = row["data_precision"], row["data_scale"]
                kind = ("NUMERIC(38, 0)" if precision == 38 and scale == 0 else
                        "SMALLINT" if precision is None and scale == 0 else
                        "NUMBER(19)" if precision == 19 and scale == 0 else kind)
            elif kind == "VARCHAR2":
                kind = f"VARCHAR2({row['char_length']} CHAR)"
            elif kind == "RAW":
                kind = f"RAW({row['data_length']})"
            elif kind.startswith("TIMESTAMP") and kind.endswith("WITH TIME ZONE"):
                kind = "TIMESTAMP WITH TIME ZONE"
            actual[row["column_name"]] = [kind, row["nullable"]]
        if actual != spec["columns"]:
            raise ValueError(f"通知表列／类型／可空性验收失败：{table}")
        pk = list((await conn.execute(text("""
            SELECT cc.column_name FROM user_constraints c JOIN user_cons_columns cc
            ON cc.constraint_name=c.constraint_name WHERE c.table_name=:name AND c.constraint_type='P'
            ORDER BY cc.position
        """), {"name": table})).scalars())
        if pk != ["ID"]:
            raise ValueError(f"通知主键验收失败：{table}")
        fks = [list(row) for row in (await conn.execute(text("""
            SELECT local_col.column_name, referenced.table_name, referenced_col.column_name
            FROM user_constraints c
            JOIN user_cons_columns local_col ON local_col.constraint_name=c.constraint_name
            JOIN user_constraints referenced ON referenced.constraint_name=c.r_constraint_name
            JOIN user_cons_columns referenced_col ON referenced_col.constraint_name=referenced.constraint_name
            AND referenced_col.position=local_col.position
            WHERE c.table_name=:name AND c.constraint_type='R' AND c.delete_rule='NO ACTION'
        """), {"name": table})).all()]
        if sorted(fks) != spec["foreign_keys"]:
            raise ValueError(f"通知外键验收失败：{table}")
    for name, (table, columns) in expected["indexes"].items():
        actual = list((await conn.execute(text("""
            SELECT ic.column_name FROM user_ind_columns ic JOIN user_indexes i
            ON i.index_name=ic.index_name WHERE ic.index_name=:name AND ic.table_name=:table_name
            AND i.uniqueness='NONUNIQUE' ORDER BY ic.column_position
        """), {"name": name, "table_name": table})).scalars())
        # Oracle 将带时区时间戳的普通索引物理化为 SYS_EXTRACT_UTC 表达式。
        # 只接受与规范列完全相同的表达式，不把任意隐藏列当作匹配。
        expressions = dict((await conn.execute(text("""
            SELECT column_position,column_expression FROM user_ind_expressions
            WHERE index_name=:name AND table_name=:table_name
        """), {"name": name, "table_name": table})).all())
        for position, column in enumerate(columns, 1):
            expression = expressions.get(position)
            if expression and str(expression).strip() == f'SYS_EXTRACT_UTC("{column}")':
                actual[position - 1] = column
        if actual != columns:
            raise ValueError(f"通知索引验收失败：{name}")
    if (await conn.execute(text("SELECT count(*) FROM KBOT_NOTIFY_THROTTLE WHERE ID='external'"))).scalar_one() != 1:
        raise ValueError("全局频控种子缺失")
    for code in ("channel_manage", "binding_manage", "delivery_read", "delivery_retry"):
        permission = "platform:notification_" + code
        count = (await conn.execute(text("""
            SELECT count(*) FROM KBOT_PERMISSION p JOIN KBOT_APP_ROLE_PERMISSION rp
            ON rp.PERMISSION_CODE=p.PERMISSION_CODE WHERE p.PERMISSION_CODE=:code
            AND p.APP_ID='platform' AND rp.APP_ID='platform' AND rp.ROLE_CODE='platform_admin'
        """), {"code": permission})).scalar_one()
        if count != 1:
            raise ValueError("通知权限或内置管理员授权缺失")


async def main(args) -> None:
    root = args.repo_root.resolve()
    os.chdir(root)
    sys.path[:0] = [str(root / "packages/platform_core/src"), str(root / "scripts/db")]
    from dotenv import load_dotenv
    from platform_core.config import get_settings
    from platform_core.database.oracle import create_database_runtime
    from apply_oracle_schema import split_oracle_statements
    load_dotenv(root / ".env", override=False)
    sql = args.sql_file.read_text(encoding="utf-8")
    expected = contract(sql)
    runtime = create_database_runtime(get_settings())
    try:
        async with runtime.engine.connect() as conn:
            if args.apply:
                for name in (*expected["tables"], *expected["indexes"]):
                    if (await conn.execute(text("SELECT count(*) FROM user_objects WHERE object_name=:name"), {"name": name})).scalar_one():
                        raise ValueError("目标对象已存在或部分升级；停止，不自动重跑")
                if (await conn.execute(text("SELECT count(*) FROM KBOT_APP_ROLE WHERE APP_ID='platform' AND ROLE_CODE='platform_admin'"))).scalar_one() != 1:
                    raise ValueError("平台管理员角色基线不符合预期")
                await conn.execute(text("ALTER SESSION SET DDL_LOCK_TIMEOUT = 5"))
                with args.report.open("x", encoding="utf-8") as report:
                    os.chmod(args.report, 0o600)
                    for number, statement in enumerate(split_oracle_statements(sql), 1):
                        cleaned = re.sub(r"--[^\n]*", "", statement).strip()
                        if not cleaned:
                            continue
                        await conn.execute(text(statement))
                        report.write(json.dumps({"step": number, "operation": cleaned.splitlines()[0], "status": "executed"}, ensure_ascii=False) + "\n")
                        report.flush()
                        os.fsync(report.fileno())
                    await conn.commit()
                    await verify(conn, expected)
                    report.write(json.dumps({"status": "verified", "tables": 8, "indexes": 7}) + "\n")
                print("Oracle 通知增量完成并验收：8表、7索引、种子及4权限；未修改已有业务表。", flush=True)
            else:
                await verify(conn, expected)
                print("Oracle 通知结构只读验收通过。", flush=True)
    finally:
        await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--sql-file", type=Path, default=DEFAULT_SQL)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("notification-delivery-oracle-upgrade.jsonl"))
    try:
        asyncio.run(main(parser.parse_args()))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "message": str(exc) if isinstance(exc, ValueError) else "部署失败，DDL 可能已部分提交；请先只读核对逐步日志，禁止盲目重跑"}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from None
