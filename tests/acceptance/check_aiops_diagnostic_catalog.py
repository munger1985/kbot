"""离线验收 AIOps 诊断目录、模板 Hash 和只读结构。"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.diagnostics import DiagnosticRegistry


def main() -> None:
    registry = DiagnosticRegistry.load()
    counts = Counter(item.definition.db_type for item in registry.tools)
    if counts["ORACLE"] != counts["MYSQL"]:
        raise RuntimeError("Oracle/MySQL 首批诊断工具数量不对等")
    print(
        "AIOps 诊断目录检查通过："
        f"tools={len(registry.tools)} "
        f"oracle={counts['ORACLE']} mysql={counts['MYSQL']} "
        f"catalog_hash={registry.catalog_hash}"
    )


if __name__ == "__main__":
    main()
