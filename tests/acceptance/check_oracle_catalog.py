"""实库验收当前 Oracle Schema 的 KBot 4.0 对象清单。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from sqlalchemy import text

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from platform_core.config import get_settings  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from tests.acceptance.check_oracle_schema import (  # noqa: E402
    SERVICE_TABLES,
    SERVICE_VIEWS,
)
from tests.support.oracle_preflight import require_oracle_listener  # noqa: E402


def expected_objects() -> dict[str, str]:
    """返回所有服务共同 Schema 中预期的对象及类型。"""
    result = {
        table: "TABLE"
        for tables in SERVICE_TABLES.values()
        for table in tables
    }
    result.update(
        {
            view: "VIEW"
            for views in SERVICE_VIEWS.values()
            for view in views
        }
    )
    return result


async def check_catalog() -> tuple[dict[str, str], list[str]]:
    """返回连接目标摘要和全部对象漂移。"""
    settings = get_settings()
    oracle = settings.database.oracle
    require_oracle_listener(host=oracle.host, port=oracle.port)
    runtime = create_database_runtime(settings)
    try:
        async with runtime.session_factory() as session:
            target = (
                await session.execute(
                    text(
                        """
                        SELECT SYS_CONTEXT('USERENV', 'CON_NAME') AS con_name,
                               SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA')
                                   AS current_schema
                        FROM dual
                        """
                    )
                )
            ).one()
            rows = (
                await session.execute(
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
    finally:
        await runtime.close()

    expected = expected_objects()
    actual = {
        str(row.object_name): str(row.object_type) for row in rows
    }
    errors: list[str] = []
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        errors.append(f"Oracle 缺少 KBot 对象：{missing}")
    if unexpected:
        errors.append(f"Oracle 出现未登记 KBot 对象：{unexpected}")
    for name in sorted(set(expected) & set(actual)):
        if expected[name] != actual[name]:
            errors.append(
                f"{name} 类型不一致："
                f"expected={expected[name]} actual={actual[name]}"
            )
    invalid = sorted(
        str(row.object_name)
        for row in rows
        if str(row.status) != "VALID"
    )
    if invalid:
        errors.append(f"Oracle 存在 INVALID KBot 对象：{invalid}")
    return {
        "container": str(target.con_name),
        "schema": str(target.current_schema),
    }, errors


def main() -> int:
    try:
        target, errors = asyncio.run(check_catalog())
    except RuntimeError as exc:
        print(f"Oracle Catalog Preflight 失败：{exc}")
        return 2
    if errors:
        print(
            "Oracle KBot 对象清单校验失败："
            f"container={target['container']} schema={target['schema']}"
        )
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Oracle KBot 对象清单校验通过："
        f"container={target['container']} schema={target['schema']} "
        f"objects={len(expected_objects())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
