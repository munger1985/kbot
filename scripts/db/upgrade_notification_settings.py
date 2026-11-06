"""定向新增通知公共设置表；保留已有渠道、凭据、通知与业务记录。"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import os
import sys

from sqlalchemy import text
from sqlalchemy.schema import CreateTable

ROOT = Path(__file__).resolve().parents[2]
ORACLE = True


async def upgrade(conn, *, table, verify_existing, verify_current, oracle, apply):
    """先验收全部既有通知结构；只创建一张缺失的新表，不回填现有数据。"""
    if not oracle:
        await conn.execute(text("SET LOCAL lock_timeout = '5s'"))
        await conn.execute(text("SET LOCAL statement_timeout = '60s'"))
    await verify_existing(conn)
    if oracle:
        exists = await conn.scalar(text("SELECT count(*) FROM user_tables WHERE table_name=:name"),
            {"name": table.name})
    else:
        exists = await conn.scalar(text("SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema=current_schema() AND table_name=:name"), {"name": table.name})
    if exists:
        await verify_current(conn)
        return False
    if not apply:
        return False
    if oracle:
        await conn.execute(text("ALTER SESSION SET DDL_LOCK_TIMEOUT = 5"))
    await conn.execute(CreateTable(table))
    await verify_current(conn)
    return True


async def main(args):
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "packages/platform_core/src"),
                   str(ROOT / "services/main_api/src")]
    from dotenv import load_dotenv
    from platform_core.config import get_settings
    load_dotenv(ROOT / ".env", override=False)
    if ORACLE:
        from platform_core.database.oracle import create_database_runtime
        from main_api.entities.notification_delivery import external_tables as tables
        from scripts.db.upgrade_notification_delivery import contract, verify
        expected = contract((ROOT / "database/oracle/main_api/003_notification_delivery.sql").read_text())
        baseline = {**expected, "tables": {k: v for k, v in expected["tables"].items()
                                          if k != tables["settings"].name}}
        runtime = create_database_runtime(get_settings())
    else:
        from platform_core.database.postgresql import create_database_runtime_from_settings
        from platform_core.persistence.core_schema import notification_external_tables as tables
        from scripts.deployment.upgrade_notification_delivery import snapshot, verify
        expected = snapshot(tables.values())
        baseline = snapshot(table for kind, table in tables.items() if kind != "settings")
        runtime = create_database_runtime_from_settings(get_settings())
    try:
        async with runtime.engine.begin() as conn:
            changed = await upgrade(conn, table=tables["settings"],
                verify_existing=lambda c: verify(c, baseline),
                verify_current=lambda c: verify(c, expected), oracle=ORACLE, apply=args.apply)
        print("已新增公共设置表，当前通知规范验收通过" if changed else
              "预检完成，未执行 DDL；缺失设置表时需显式 --apply")
        print("没有删除、回填或改写既有渠道、凭据、任务及业务数据")
    finally:
        await runtime.engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="预检通过后仅新增公共设置表")
    try:
        asyncio.run(main(parser.parse_args()))
    except Exception as exc:
        print("通知公共设置升级失败：" + type(exc).__name__ + "；未输出 SQL 参数或凭据，请只读核对结构")
        raise SystemExit(1) from None
