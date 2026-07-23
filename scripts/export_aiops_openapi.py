"""导出 AIOps Public、Internal 与 Executor OpenAPI 快照。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.bootstrap.openapi import (  # noqa: E402
    create_executor_contract_app,
    create_internal_contract_app,
    create_public_contract_app,
)


def export_openapi(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apps = {
        "aiops_public_v1.json": create_public_contract_app(),
        "aiops_internal_v1.json": create_internal_contract_app(),
        "aiops_executor_v1.json": create_executor_contract_app(),
    }
    for filename, app in apps.items():
        output = output_dir / filename
        output.write_text(
            json.dumps(
                app.openapi(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"已导出 {output.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="导出 AIOps OpenAPI 快照")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "openapi",
    )
    args = parser.parse_args()
    export_openapi(args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
