"""Static dependency guard for the active 4.0 service packages.

This intentionally does not inspect ``legacy/``.  Legacy code may still refer
to the 3.x repository/controller graph while it is being retired, but active
Knowledge Core, model-serving and platform packages must remain independently
deployable.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (
    ROOT / "platform_core",
    ROOT / "platform_clients",
    ROOT / "knowledge_core",
    ROOT / "model_serving",
    ROOT / "apps" / "knowledge_core_api",
    ROOT / "apps" / "knowledge_core_parser",
    ROOT / "apps" / "knowledge_core_projection",
    ROOT / "apps" / "ai_models_embedding",
    ROOT / "apps" / "ai_models_llm",
    ROOT / "apps" / "ai_models_vlm",
    ROOT / "apps" / "ai_models_visual",
)
FORBIDDEN_PREFIXES = (
    "dao",
    "services",
    "agent",
    "skills",
    "microservices",
)


def module_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        imported = module_name(node)
        if not imported:
            continue
        if imported == "platform_core.auth" or imported.startswith("platform_core.auth."):
            violations.append(f"{path}:{node.lineno}: legacy user-auth import {imported}")
        if imported == "utils" or imported.startswith("utils."):
            violations.append(f"{path}:{node.lineno}: legacy utility import {imported}")
            continue
        if imported == "legacy" or imported.startswith("legacy."):
            violations.append(f"{path}:{node.lineno}: legacy package import {imported}")
            continue
        if imported == "" or not imported.split(".")[0] in FORBIDDEN_PREFIXES:
            continue
        violations.append(f"{path}:{node.lineno}: forbidden cross-domain import {imported}")
    return violations


def main() -> int:
    violations: list[str] = []
    for root in ACTIVE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" not in path.parts:
                violations.extend(check_file(path))
    if violations:
        print("4.0 boundary violations:")
        print("\n".join(violations))
        return 1
    print("4.0 boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
