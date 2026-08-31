"""离线验收 AIOps 诊断目录、模板 Hash 和只读结构。"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.diagnostics import DiagnosticRegistry


REQUIRED_TOOLS = {
    "ORACLE": {
        "db.instance.identity",
        "db.sql.top_current",
        "db.session.active",
        "db.session.blocking_chain",
        "db.storage.capacity",
        "db.instance.performance",
        "db.memory.summary",
        "db.resource.session_utilization",
        "db.wait.class_summary",
        "db.archive.status",
        "db.transaction.long_running",
        "db.replication.status",
        "db.instance.parameters",
        "db.storage.temp_usage",
        "db.storage.undo_usage",
        "db.redo.status",
        "db.alert.recent",
        "db.scheduler.failed_jobs",
        "db.objects.invalid_summary",
        "db.backup.recent_jobs",
        "db.replication.lag",
    },
    "MYSQL": {
        "db.instance.identity",
        "db.session.active",
        "db.session.blocking_chain",
        "db.storage.capacity",
        "db.transaction.long_running",
        "db.replication.status",
    },
    "POSTGRESQL": {
        "db.instance.identity",
        "db.session.active",
        "db.session.blocking_chain",
        "db.storage.capacity",
        "db.transaction.long_running",
        "db.replication.status",
    },
}


def main() -> None:
    registry = DiagnosticRegistry.load()
    counts = Counter(item.definition.db_type for item in registry.tools)
    tool_ids: dict[str, set[str]] = defaultdict(set)
    for item in registry.tools:
        tool_ids[item.definition.db_type].add(item.definition.tool_id)
    missing = {
        db_type: sorted(required - tool_ids[db_type])
        for db_type, required in REQUIRED_TOOLS.items()
        if required - tool_ids[db_type]
    }
    if missing:
        detail = "; ".join(
            f"{db_type}={','.join(tool_names)}"
            for db_type, tool_names in sorted(missing.items())
        )
        raise RuntimeError(f"AIOps 诊断目录缺少必备工具：{detail}")
    print(
        "AIOps 诊断目录检查通过："
        f"tools={len(registry.tools)} "
        f"oracle={counts['ORACLE']} mysql={counts['MYSQL']} "
        f"postgresql={counts['POSTGRESQL']} "
        f"catalog_hash={registry.catalog_hash}"
    )


if __name__ == "__main__":
    main()
