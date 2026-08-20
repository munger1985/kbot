"""将仓库 Prompt Catalog 同步到现有 Oracle Schema。"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

from platform_core.config import get_settings
from platform_core.database.oracle import create_database_runtime
from platform_core.prompts import load_prompt_catalog, sync_prompt_catalog


REQUIRED_TABLES = {
    "KBOT_PLATFORM_PROMPT",
    "KBOT_PLATFORM_PROMPT_VERSION",
}


async def synchronize(*, selected_services: set[str]) -> tuple[int, str, str, str]:
    """同步指定服务并返回数量、Catalog Hash、PDB 和 Schema。"""
    catalog = load_prompt_catalog()
    settings = get_settings()
    runtime = create_database_runtime(settings)
    try:
        async with runtime.engine.connect() as connection:
            target = (
                await connection.execute(
                    text(
                        """
                        SELECT SYS_CONTEXT('USERENV', 'CON_NAME'),
                               SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                        FROM DUAL
                        """
                    )
                )
            ).one()
            existing_tables = set(
                (
                    await connection.execute(
                        text(
                            """
                            SELECT table_name
                            FROM user_tables
                            WHERE table_name IN (
                                'KBOT_PLATFORM_PROMPT',
                                'KBOT_PLATFORM_PROMPT_VERSION'
                            )
                            """
                        )
                    )
                ).scalars().all()
            )
            missing = sorted(REQUIRED_TABLES - existing_tables)
            if missing:
                raise RuntimeError(
                    "当前 Schema 缺少 Prompt Catalog 表："
                    f"{', '.join(missing)}"
                )
            count = await sync_prompt_catalog(
                connection,
                selected_services=selected_services,
                environment=settings.environment,
            )
            return count, catalog.catalog_sha256, str(target[0]), str(target[1])
    finally:
        await runtime.close()


def main() -> int:
    catalog = load_prompt_catalog()
    available_services = {
        entry.owner_service
        for entry in catalog.entries
        if entry.owner_service != "platform"
    }
    parser = argparse.ArgumentParser(
        description="将仓库 Prompt Catalog 幂等同步到现有 Oracle Schema"
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        help="只同步指定 owner_service；可重复，默认同步全部服务",
    )
    args = parser.parse_args()
    selected_services = set(args.service) or available_services
    unknown = sorted(selected_services - available_services)
    if unknown:
        print(f"Prompt Catalog 同步失败：未知服务：{', '.join(unknown)}")
        return 1
    try:
        count, catalog_hash, pdb_name, schema_name = asyncio.run(
            synchronize(selected_services=selected_services)
        )
    except Exception as exc:
        print(f"Prompt Catalog 同步失败：{exc}")
        return 1
    print(
        "Prompt Catalog 同步完成："
        f"PDB={pdb_name}，Schema={schema_name}，"
        f"Prompt={count}，Catalog Hash={catalog_hash}，"
        f"Environment={get_settings().environment}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
