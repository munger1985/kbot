"""为现有 Oracle Schema 安全补建并核验 S6 通知表。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text

from platform_core.database.oracle import create_database_runtime
from apply_oracle_schema import split_oracle_statements


ROOT = Path(__file__).resolve().parents[2]
DDL_PATH = ROOT / "database/oracle/platform_core/003_notifications.sql"
TABLES = {
    "KBOT_NOTIFICATION_OUTBOX",
    "KBOT_NOTIFICATION_INBOX",
    "KBOT_NOTIFICATION_PREF",
    "KBOT_WORK_ITEM",
    "KBOT_BACKGROUND_OPERATION",
    "KBOT_OPERATION_WATCH",
}


async def main() -> None:
    runtime = create_database_runtime()
    try:
        async with runtime.engine.begin() as connection:
            existing = set((await connection.execute(text(
                "SELECT TABLE_NAME FROM USER_TABLES WHERE TABLE_NAME IN "
                "('KBOT_NOTIFICATION_OUTBOX','KBOT_NOTIFICATION_INBOX',"
                "'KBOT_NOTIFICATION_PREF','KBOT_WORK_ITEM',"
                "'KBOT_BACKGROUND_OPERATION','KBOT_OPERATION_WATCH')"
            ))).scalars().all())
            if existing and existing != TABLES:
                missing = ", ".join(sorted(TABLES - existing))
                raise RuntimeError(f"通知 Schema 处于部分创建状态，缺少：{missing}")
            if not existing:
                for statement in split_oracle_statements(
                    DDL_PATH.read_text(encoding="utf-8")
                ):
                    await connection.execute(text(statement))
                print("S6 通知 Schema 已创建")
            else:
                print("S6 通知 Schema 已存在，跳过创建")
            counts = dict((await connection.execute(text(
                "SELECT TABLE_NAME, COUNT(*) OVER () AS TABLE_COUNT "
                "FROM USER_TABLES WHERE TABLE_NAME IN "
                "('KBOT_NOTIFICATION_OUTBOX','KBOT_NOTIFICATION_INBOX',"
                "'KBOT_NOTIFICATION_PREF','KBOT_WORK_ITEM',"
                "'KBOT_BACKGROUND_OPERATION','KBOT_OPERATION_WATCH')"
            ))).all())
            if set(counts) != TABLES or set(counts.values()) != {6}:
                raise RuntimeError("通知 Schema 核验失败")
            print("S6 通知 Schema 核验通过：6 张表")
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
