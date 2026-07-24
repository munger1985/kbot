"""核对服务 Entity 与规范 Oracle 表所有权。"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent_runtime.entities as agent_entities  # noqa: E402
import aiops_agent.entities as aiops_entities  # noqa: E402
import knowledge_core.entities as kc_entities  # noqa: E402
import main_api.entities as platform_entities  # noqa: E402
import model_serving.common.entities as model_entities  # noqa: E402
from scripts.check_oracle_schema import SERVICE_TABLES  # noqa: E402


ENTITY_MODULES = {
    "platform_core": platform_entities,
    "model_serving": model_entities,
    "knowledge_core": kc_entities,
    "agent_runtime": agent_entities,
    "aiops_agent": aiops_entities,
}


def entity_tables_by_service() -> dict[str, set[str]]:
    """读取每个服务显式导出的 Entity 表名。"""
    return {
        service: {
            str(getattr(module, name).__tablename__).upper()
            for name in module.__all__
            if hasattr(getattr(module, name), "__tablename__")
        }
        for service, module in ENTITY_MODULES.items()
    }


def check_entity_ownership() -> list[str]:
    """返回 Entity 缺失、越界或重复映射错误。"""
    mapped = entity_tables_by_service()
    errors: list[str] = []
    owners: dict[str, str] = {}
    for service, expected in SERVICE_TABLES.items():
        actual = mapped.get(service, set())
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing:
            errors.append(f"{service} 缺少 Entity：{missing}")
        if unexpected:
            errors.append(f"{service} 出现越界 Entity：{unexpected}")
        for table in actual:
            previous = owners.setdefault(table, service)
            if previous != service:
                errors.append(
                    f"{table} 被 {previous} 和 {service} 重复映射"
                )
    return errors


def main() -> int:
    errors = check_entity_ownership()
    if errors:
        print("KBot Entity 所有权校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    count = sum(len(value) for value in entity_tables_by_service().values())
    print(f"KBot Entity 所有权校验通过：5 个服务，{count} 张表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
