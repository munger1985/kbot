"""Audit shared AIOps contracts and catalog identifiers across both repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog_ids(root: Path, relative_glob: str, collection: str, key: str) -> list[str]:
    identifiers: set[str] = set()
    for path in root.glob(relative_glob):
        identifiers.update(str(item[key]) for item in _json(path)[collection])
    return sorted(identifiers)


def _inspection_ids(root: Path) -> list[str]:
    path = (
        root
        / "services/aiops_agent/src/aiops_agent/application/inspections/check_catalog.json"
    )
    return sorted(
        str(check["check_id"])
        for group in _json(path)["groups"]
        for check in group["checks"]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> dict[str, Any]:
    return {
        "actions": _catalog_ids(
            root,
            "services/aiops_agent/src/aiops_agent/actions/catalog/*/manifest.json",
            "actions",
            "action_template_id",
        ),
        "tools": _catalog_ids(
            root,
            "services/aiops_agent/src/aiops_agent/diagnostics/catalog/*/manifest.json",
            "tools",
            "tool_id",
        ),
        "checks": _inspection_ids(root),
        "metrics": _catalog_ids(
            root,
            "services/aiops_agent/src/aiops_agent/resources/metrics/*.json",
            "metrics",
            "metric_code",
        ),
        "profiles": _catalog_ids(
            root,
            "services/aiops_agent/src/aiops_agent/resources/monitoring/*.json",
            "profiles",
            "profile_id",
        ),
    }


def _default_peer(root: Path) -> Path:
    peer_name = "kbot4" if root.name == "ammolite_cube" else "ammolite_cube"
    return root.parent / peer_name


def audit(root: Path, peer: Path) -> dict[str, Any]:
    local = _inventory(root)
    remote = _inventory(peer)
    comparisons = {
        name: {
            "classification": "DIRECT_COPY",
            "equal": local[name] == remote[name],
            "local_count": len(local[name]),
            "peer_count": len(remote[name]),
            "local_only": sorted(set(local[name]) - set(remote[name])),
            "peer_only": sorted(set(remote[name]) - set(local[name])),
        }
        for name in local
    }
    plan = Path("docs/proposals/aiops-cross-project-unified-implementation-plan.md")
    plan_equal = _sha256(root / plan) == _sha256(peer / plan)
    local_agents = (
        root / "services/aiops_agent/src/aiops_agent/application/agents.py"
    ).read_text(encoding="utf-8")
    peer_agents = (
        peer / "services/aiops_agent/src/aiops_agent/application/agents.py"
    ).read_text(encoding="utf-8")
    semantic_markers = {
        "explicit_controlled_actions": all(
            marker in source
            for source in (local_agents, peer_agents)
            for marker in (
                "TargetControlledActionExecution",
                "AIOPS_AGENT_ACTION_SCOPE_REQUIRED",
            )
        ),
        "complete_target_source_matrix": all(
            marker in source
            for source in (local_agents, peer_agents)
            for marker in (
                "list_source_bindings",
                "AIOPS_AGENT_SOURCE_BINDING_REQUIRED",
                "AIOPS_AGENT_SOURCE_UNAVAILABLE",
            )
        ),
    }
    passed = (
        all(item["equal"] for item in comparisons.values())
        and plan_equal
        and all(semantic_markers.values())
    )
    return {
        "schema_version": "aiops.cross-project-alignment.v1",
        "local_root": str(root),
        "peer_root": str(peer),
        "passed": passed,
        "comparisons": comparisons,
        "mirrored_plan": {
            "classification": "DIRECT_COPY",
            "equal": plan_equal,
        },
        "semantic_markers": semantic_markers,
        "allowed_adaptations": [
            "META_DATABASE_ADAPTATION",
            "FRONTEND_ADAPTATION",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-root", type=Path, default=_default_peer(ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit(ROOT, args.peer_root.resolve())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for name, comparison in result["comparisons"].items():
            state = "PASS" if comparison["equal"] else "FAIL"
            print(
                f"{state} {name}: "
                f"{comparison['local_count']} / {comparison['peer_count']}"
            )
        for name, passed in result["semantic_markers"].items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")
        print(
            f"{'PASS' if result['mirrored_plan']['equal'] else 'FAIL'} "
            "mirrored_plan"
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
