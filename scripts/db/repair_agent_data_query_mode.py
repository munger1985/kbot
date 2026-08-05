"""将既有 mcp_data Agent 一次性切换到 4.0 data_query/MCP 配置。"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from platform_core.database.oracle import create_database_runtime


async def repair(*, apply: bool) -> None:
    runtime = create_database_runtime()
    try:
        async with runtime.engine.begin() as connection:
            column_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM USER_TAB_COLUMNS "
                "WHERE TABLE_NAME='KBOT_AGENT_DEFINITION' "
                "AND COLUMN_NAME='DATA_QUERY_MODE'"
            ))).scalar_one())
            if column_count != 1:
                raise RuntimeError(
                    "缺少 DATA_QUERY_MODE；请先按 Canonical DDL 对齐 Agent Runtime Schema"
                )
            legacy_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM KBOT_AGENT_DEFINITION "
                "WHERE JSON_EXISTS(ENABLED_CAPABILITIES_JSON, '$?(@ == \"mcp_data\")')"
            ))).scalar_one())
            invalid_count = int((await connection.execute(text(
                "SELECT COUNT(*) FROM KBOT_AGENT_DEFINITION "
                "WHERE JSON_EXISTS(ENABLED_CAPABILITIES_JSON, '$?(@ == \"mcp_data\")') "
                "AND (DATA_PROFILE_NAME IS NULL OR TRIM(DATA_PROFILE_NAME) IS NULL)"
            ))).scalar_one())
            if invalid_count:
                raise RuntimeError(
                    f"发现 {invalid_count} 个旧问数 Agent 缺少 data_profile_name，拒绝修复"
                )
            print(f"待修复旧问数 Agent：{legacy_count}")
            if not apply or legacy_count == 0:
                return
            result = await connection.execute(text(
                "UPDATE KBOT_AGENT_DEFINITION SET "
                "DATA_QUERY_MODE='MCP', "
                "ENABLED_CAPABILITIES_JSON=JSON(REPLACE(JSON_SERIALIZE("
                "ENABLED_CAPABILITIES_JSON RETURNING CLOB), "
                "'\"mcp_data\"', '\"data_query\"')), "
                "ROW_VERSION=ROW_VERSION+1, UPDATED_AT=CURRENT_TIMESTAMP "
                "WHERE JSON_EXISTS(ENABLED_CAPABILITIES_JSON, '$?(@ == \"mcp_data\")')"
            ))
            if int(result.rowcount or 0) != legacy_count:
                raise RuntimeError("修复行数与预检行数不一致，事务已回滚")
            print(f"已将 {legacy_count} 个 Agent 切换为 data_query/MCP")
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="执行修复；缺省仅做预检并打印数量",
    )
    args = parser.parse_args()
    asyncio.run(repair(apply=args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
