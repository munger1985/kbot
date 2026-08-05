"""生成并验收 KBot 4.0 全部 HTTP 服务的 OpenAPI 快照。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPENAPI_ROOT = ROOT / "docs" / "openapi"


def build_contracts() -> dict[str, dict[str, Any]]:
    """创建不启动 Lifespan、不连接数据库的契约快照。"""
    from aiops_agent.bootstrap.openapi import (  # noqa: PLC0415
        create_executor_contract_app,
        create_internal_contract_app,
    )
    from agent_runtime.entrypoints.api import (  # noqa: PLC0415
        app as agent_runtime_app,
    )
    from data_query.bootstrap.openapi import (  # noqa: PLC0415
        create_data_query_contract_app,
    )
    from model_serving.entrypoints.embedding import (  # noqa: PLC0415
        app as embedding_app,
    )
    from model_serving.entrypoints.llm import app as llm_app  # noqa: PLC0415
    from model_serving.entrypoints.visual import (  # noqa: PLC0415
        app as visual_app,
    )
    from model_serving.entrypoints.vlm import app as vlm_app  # noqa: PLC0415
    from knowledge_core.entrypoints.api import (  # noqa: PLC0415
        app as knowledge_core_app,
    )
    from main_api.app import create_main_api_app  # noqa: PLC0415
    from main_api.openapi_contracts import (  # noqa: PLC0415
        create_aiops_public_contract_app,
    )

    apps = {
        "main_api_public_v1.json": create_main_api_app(
            enable_access_log=False
        ),
        "knowledge_core_internal_v1.json": knowledge_core_app,
        "agent_runtime_internal_v1.json": agent_runtime_app,
        "data_query_internal_v1.json": create_data_query_contract_app(),
        "model_embedding_v1.json": embedding_app,
        "model_llm_v1.json": llm_app,
        "model_visual_v1.json": visual_app,
        "model_vlm_v1.json": vlm_app,
        "aiops_public_v1.json": create_aiops_public_contract_app(),
        "aiops_internal_v1.json": create_internal_contract_app(),
        "aiops_executor_v1.json": create_executor_contract_app(),
    }
    return {filename: app.openapi() for filename, app in apps.items()}


def _route_boundary_errors(
    filename: str,
    schema: dict[str, Any],
) -> list[str]:
    paths = set(schema.get("paths") or {})
    errors: list[str] = []
    if filename == "main_api_public_v1.json":
        invalid = sorted(
            path
            for path in paths
            if not path.startswith("/api/v1")
            and path not in {"/healthz", "/readyz"}
        )
        if invalid:
            errors.append(f"{filename} 暴露非公开路径：{invalid}")
    elif filename in {
        "knowledge_core_internal_v1.json",
        "agent_runtime_internal_v1.json",
        "data_query_internal_v1.json",
        "aiops_internal_v1.json",
        "aiops_executor_v1.json",
    }:
        invalid = sorted(
            path
            for path in paths
            if not path.startswith("/internal/v1")
            and path not in {"/health", "/healthz", "/readyz"}
        )
        if invalid:
            errors.append(f"{filename} 暴露非内部路径：{invalid}")
    elif filename == "aiops_public_v1.json":
        invalid = sorted(
            path for path in paths if not path.startswith("/api/v1")
        )
        if invalid:
            errors.append(f"{filename} 暴露非公开路径：{invalid}")
    return errors


def check_openapi_contracts() -> list[str]:
    expected = build_contracts()
    errors: list[str] = []
    stored_files = {path.name for path in OPENAPI_ROOT.glob("*.json")}
    expected_files = set(expected)
    if stored_files != expected_files:
        errors.append(
            "OpenAPI 快照集合不一致："
            f"缺少={sorted(expected_files - stored_files)} "
            f"多余={sorted(stored_files - expected_files)}"
        )
    for filename, schema in expected.items():
        errors.extend(_route_boundary_errors(filename, schema))
        path = OPENAPI_ROOT / filename
        if not path.is_file():
            continue
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{filename} 无法读取：{exc}")
            continue
        if stored != schema:
            errors.append(f"{filename} 与当前代码契约不一致")
    return errors


def write_openapi_contracts() -> None:
    OPENAPI_ROOT.mkdir(parents=True, exist_ok=True)
    contracts = build_contracts()
    for existing in OPENAPI_ROOT.glob("*.json"):
        if existing.name not in contracts:
            raise RuntimeError(f"存在未管理快照，拒绝删除：{existing.name}")
    for filename, schema in contracts.items():
        output = OPENAPI_ROOT / filename
        output.write_text(
            json.dumps(
                schema,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"已写入 OpenAPI 快照：{output.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="根据当前代码重建全部受管理 OpenAPI 快照",
    )
    args = parser.parse_args()
    if args.write:
        write_openapi_contracts()
    errors = check_openapi_contracts()
    if errors:
        print("KBot OpenAPI 契约校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    contracts = build_contracts()
    route_count = sum(
        len(schema.get("paths") or {}) for schema in contracts.values()
    )
    print(
        "KBot OpenAPI 契约校验通过："
        f"{len(contracts)} 个快照，{route_count} 条路径"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
