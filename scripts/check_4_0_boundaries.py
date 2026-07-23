"""KBot 4.0 活跃服务包的静态依赖边界检查。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (
    ROOT / "platform_core",
    ROOT / "platform_clients",
    ROOT / "knowledge_core",
    ROOT / "model_serving",
    ROOT / "agent_runtime",
    ROOT / "aiops",
    ROOT / "apps",
)
OBSOLETE_PATHS = (
    ROOT / "legacy",
    ROOT / "agent",
    ROOT / "skills",
    ROOT / "microservices",
    ROOT / "utils",
)
FORBIDDEN_PREFIXES = (
    "dao",
    "services",
    "agent",
    "skills",
    "microservices",
    "utils",
    "legacy",
)
OBSOLETE_API_PREFIXES = (
    "/v4",
    "/api/v2",
    "/internal/v2",
)


def module_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module else []
    return []


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for prefix in OBSOLETE_API_PREFIXES:
                if node.value == prefix or node.value.startswith(f"{prefix}/"):
                    violations.append(
                        f"{path}:{node.lineno}: 禁止使用旧 API 前缀 {prefix}"
                    )
        for imported in module_names(node):
            if imported == "platform_core.auth" or imported.startswith("platform_core.auth."):
                violations.append(f"{path}:{node.lineno}: 禁止导入旧用户认证模块 {imported}")
                continue
            if imported.split(".")[0] in FORBIDDEN_PREFIXES:
                violations.append(f"{path}:{node.lineno}: 禁止导入旧模块或跨领域模块 {imported}")
    return violations


def main() -> int:
    violations: list[str] = []
    for path in OBSOLETE_PATHS:
        if path.exists():
            violations.append(f"{path}: 4.0 工作树中禁止保留旧代码目录")
    for root in ACTIVE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" not in path.parts:
                violations.extend(check_file(path))
    if violations:
        print("发现 KBot 4.0 架构边界违规：")
        print("\n".join(violations))
        return 1
    print("KBot 4.0 架构边界检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
