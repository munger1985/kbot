"""为现有 Oracle Schema 安全补建并核验 S7 组合回执表。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text

from platform_core.database.oracle import create_database_runtime
from apply_oracle_schema import split_oracle_statements


ROOT = Path(__file__).resolve().parents[2]
DDL_PATH = ROOT / "database/oracle/platform_core/004_composition.sql"
TABLE = "KBOT_COMPOSITION_RECEIPT"


async def main() -> None:
    runtime = create_database_runtime()
    try:
        async with runtime.engine.begin() as connection:
            exists = int((await connection.execute(text(
                "SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME = "
                "'KBOT_COMPOSITION_RECEIPT'"
            ))).scalar_one())
            if not exists:
                for statement in split_oracle_statements(
                    DDL_PATH.read_text(encoding="utf-8")
                ):
                    await connection.execute(text(statement))
                print("S7 组合回执 Schema 已创建")
            else:
                print("S7 组合回执 Schema 已存在，跳过创建")
            columns = set((await connection.execute(text(
                "SELECT COLUMN_NAME FROM USER_TAB_COLUMNS WHERE TABLE_NAME = "
                "'KBOT_COMPOSITION_RECEIPT'"
            ))).scalars().all())
            required = {
                "RECEIPT_ID", "DOMAIN_ID", "ACTOR_ID", "OPERATION",
                "IDEMPOTENCY_KEY", "REQUEST_HASH", "STATUS",
                "RESOURCE_TYPE", "RESOURCE_ID", "VERIFICATION_JSON",
                "ATTEMPT_COUNT", "ROW_VERSION", "CREATED_AT", "UPDATED_AT",
            }
            if not required.issubset(columns):
                raise RuntimeError(
                    "组合回执 Schema 核验失败，缺少列："
                    + ", ".join(sorted(required - columns))
                )
            print(f"S7 组合回执 Schema 核验通过：{TABLE}")
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
