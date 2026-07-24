"""核对五个服务全部 SQLAlchemy Entity 与当前 Oracle 列契约。"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from sqlalchemy import bindparam, text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from platform_core.config import get_settings  # noqa: E402
from platform_core.database.oracle import create_database_runtime  # noqa: E402
from scripts.check_aiops_entity_schema import (  # noqa: E402
    ColumnContract,
    catalog_column_contract,
    entity_column_contract,
)
from scripts.check_entity_ownership import ENTITY_MODULES  # noqa: E402
from scripts.oracle_preflight import require_oracle_listener  # noqa: E402


def _entity_classes() -> tuple[type, ...]:
    return tuple(
        entity
        for modules in ENTITY_MODULES.values()
        for module in (
            modules if isinstance(modules, tuple) else (modules,)
        )
        for name in module.__all__
        if hasattr((entity := getattr(module, name)), "__table__")
    )


def _compatible(
    expected: ColumnContract,
    actual: ColumnContract,
) -> bool:
    """无显式精度的通用类型只比较其声明过的物理属性。"""
    return (
        expected.family == actual.family
        and expected.nullable == actual.nullable
        and (
            expected.length is None
            or expected.length == actual.length
        )
        and (
            expected.precision is None
            or expected.precision == actual.precision
        )
        and (
            expected.scale is None
            or expected.scale == actual.scale
        )
        and (
            expected.timezone is None
            or expected.timezone == actual.timezone
        )
    )


async def check_all_entity_schema() -> list[str]:
    settings = get_settings()
    oracle = settings.database.oracle
    require_oracle_listener(host=oracle.host, port=oracle.port)
    runtime = create_database_runtime(settings)
    entities = _entity_classes()
    table_names = tuple(entity.__tablename__ for entity in entities)
    try:
        async with runtime.session_factory() as session:
            rows = (
                await session.execute(
                    text(
                        """
                        SELECT table_name, column_name, data_type, data_length,
                               data_precision, data_scale, char_length, nullable
                        FROM user_tab_columns
                        WHERE table_name IN :table_names
                        ORDER BY table_name, column_id
                        """
                    ).bindparams(
                        bindparam(
                            "table_names",
                            value=table_names,
                            expanding=True,
                        )
                    )
                )
            ).all()
    finally:
        await runtime.close()

    catalog = {
        (row.table_name, row.column_name): catalog_column_contract(row)
        for row in rows
    }
    expected = {
        (entity.__tablename__, column.name.upper()): entity_column_contract(
            column
        )
        for entity in entities
        for column in entity.__table__.columns
    }
    errors: list[str] = []
    missing = sorted(set(expected) - set(catalog))
    unexpected = sorted(set(catalog) - set(expected))
    if missing:
        errors.append(f"Oracle 缺少 Entity 列：{missing}")
    if unexpected:
        errors.append(f"Oracle 出现未映射列：{unexpected}")
    for key in sorted(set(expected) & set(catalog)):
        if not _compatible(expected[key], catalog[key]):
            errors.append(
                f"{key[0]}.{key[1]} 不一致："
                f"entity={expected[key]} oracle={catalog[key]}"
            )
    return errors


def main() -> int:
    try:
        errors = asyncio.run(check_all_entity_schema())
    except RuntimeError as exc:
        print(f"全服务 Entity/Oracle Schema Preflight 失败：{exc}")
        return 2
    if errors:
        print("全服务 Entity/Oracle Schema 校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    entities = _entity_classes()
    columns = sum(len(entity.__table__.columns) for entity in entities)
    print(
        "全服务 Entity/Oracle Schema 校验通过："
        f"{len(entities)} 张表，{columns} 列"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
