"""KBot 4.0 活跃服务包的静态依赖边界验收。"""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ROOTS = (
    ROOT / "packages" / "platform_core" / "src" / "platform_core",
    ROOT / "packages" / "platform_clients" / "src" / "platform_clients",
    ROOT / "services" / "knowledge_core" / "src" / "knowledge_core",
    ROOT / "services" / "model_serving" / "src" / "model_serving",
    ROOT / "services" / "main_api" / "src" / "main_api",
    ROOT / "services" / "agent_runtime" / "src" / "agent_runtime",
    ROOT / "services" / "aiops_agent" / "src" / "aiops_agent",
)
OBSOLETE_PATHS = (
    ROOT / "legacy",
    ROOT / "agent",
    ROOT / "skills",
    ROOT / "microservices",
    ROOT / "utils",
    ROOT / "migrations",
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
SERVICE_PACKAGES = {
    "main_api",
    "agent_runtime",
    "knowledge_core",
    "aiops_agent",
    "model_serving",
}
RUNTIME_ROOTS = (ROOT / "packages", ROOT / "services")
EXTERNAL_DATABASE_SCHEMA_PATHS = (
    "packages/platform_core/src/platform_core/managed_database/"
    "connection_tester.py",
    "services/data_query/src/data_query/connectors/",
    "services/aiops_agent/src/aiops_agent/application/configuration/"
    "connection_test.py",
    "services/aiops_agent/src/aiops_agent/adapters/target_connection.py",
    "services/aiops_agent/src/aiops_agent/executor/drivers/",
    "services/aiops_agent/src/aiops_agent/diagnostics/",
)
FORBIDDEN_RUNTIME_SCHEMA_MARKERS = (
    "schema_ready_check",
    "check_aiops_schema",
    "agent_schema_is_ready",
    "aiops_schema_is_ready",
    "AIOPS_SCHEMA_NOT_READY",
    "_aiops_schema_manifest",
)
DATABASE_CATALOG_MARKERS = (
    "INFORMATION_SCHEMA.COLUMNS",
    "INFORMATION_SCHEMA.TABLES",
    "INFORMATION_SCHEMA.TABLE_CONSTRAINTS",
    "INFORMATION_SCHEMA.CHECK_CONSTRAINTS",
    "INFORMATION_SCHEMA.STATISTICS",
    "PG_CATALOG.PG_ATTRIBUTE",
    "PG_CATALOG.PG_CLASS",
    "PG_CATALOG.PG_CONSTRAINT",
    "PG_CATALOG.PG_INDEX",
    "PG_CATALOG.PG_INDEXES",
    "USER_TAB_COLUMNS",
    "USER_TABLES",
    "USER_CONSTRAINTS",
    "ALL_TAB_COLUMNS",
    "ALL_CONSTRAINTS",
    "DBA_TAB_COLUMNS",
    "DBA_TABLES",
    "DBA_CONSTRAINTS",
)
SELF_SCHEMA_PROBE = re.compile(
    r"SELECT\s+1\s+FROM\s+(?:KBOT_|AIOPS_|AGENT_|AMMOLITE_)"
    r"[A-Z0-9_]*\s+WHERE\s+1\s*=\s*0",
)
SELF_SCHEMA_VERSION_VIEW = re.compile(
    r"\b(?:KBOT|AMMOLITE)_V_[A-Z0-9_]*SCHEMA_VERSION\b",
)


def module_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return []
        return [node.module] if node.module else []
    return []


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]
    violations: list[str] = []
    relative_path = path.relative_to(ROOT)
    is_main_api = "main_api" in relative_path.parts
    is_aiops = "aiops_agent" in relative_path.parts
    is_aiops_domain = (
        "aiops_agent" in relative_path.parts
        and "domain" in relative_path.parts
    )
    owner_service = next(
        (
            service
            for service in SERVICE_PACKAGES
            if service in relative_path.parts
        ),
        None,
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for prefix in OBSOLETE_API_PREFIXES:
                if node.value == prefix or node.value.startswith(f"{prefix}/"):
                    violations.append(
                        f"{path}:{node.lineno}: 禁止使用旧 API 前缀 {prefix}"
                    )
        for imported in module_names(node):
            imported_root = imported.split(".")[0]
            if (
                owner_service
                and imported_root in SERVICE_PACKAGES
                and imported_root != owner_service
            ):
                violations.append(
                    f"{path}:{node.lineno}: 服务 {owner_service} 禁止直接导入"
                    f" {imported}，必须通过 platform_clients 与稳定契约调用"
                )
                continue
            if is_aiops_domain and imported.split(".")[0] in {
                "aiohttp",
                "fastapi",
                "loguru",
                "platform_clients",
                "platform_core",
                "pydantic",
                "sqlalchemy",
            }:
                violations.append(
                    f"{path}:{node.lineno}: AIOps Domain 只能依赖标准库，"
                    f"禁止导入 {imported}"
                )
                continue
            if is_aiops and (
                imported.startswith("knowledge_core.entities")
                or imported.startswith("knowledge_core.repositories")
                or imported.startswith("knowledge_core.persistence")
                or imported.startswith("model_serving.common.entities")
                or imported.startswith("model_serving.common.model_repository")
                or imported.startswith("agent_runtime.entities")
                or imported.startswith("agent_runtime.repositories")
                or imported.startswith("agent_runtime.persistence")
            ):
                violations.append(
                    f"{path}:{node.lineno}: AIOps 禁止导入其他领域持久化模块 "
                    f"{imported}"
                )
                continue
            if is_main_api and imported.split(".")[0] in {
                "knowledge_core",
                "model_serving",
            }:
                violations.append(
                    f"{path}:{node.lineno}: Main API 必须通过 platform_clients 调用内部服务，"
                    f"禁止直接导入 {imported}"
                )
                continue
            if (
                imported == "platform_core.auth"
                or imported.startswith("platform_core.auth.")
                or imported == "platform_core.platform.security"
                or imported.startswith("platform_core.platform.security.")
            ):
                violations.append(f"{path}:{node.lineno}: 禁止导入旧用户认证模块 {imported}")
                continue
            if imported.split(".")[0] in FORBIDDEN_PREFIXES:
                violations.append(f"{path}:{node.lineno}: 禁止导入旧模块或跨领域模块 {imported}")
    return violations


def check_runtime_database_boundary(path: Path) -> list[str]:
    """禁止运行时代码主动核对应用自身数据库 Schema。"""
    source = path.read_text(encoding="utf-8")
    violations: list[str] = []
    lowered = source.lower()
    for marker in FORBIDDEN_RUNTIME_SCHEMA_MARKERS:
        offset = lowered.find(marker.lower())
        if offset >= 0:
            line = source.count("\n", 0, offset) + 1
            violations.append(
                f"{path}:{line}: 运行时代码禁止 Schema 门禁标记 {marker}"
            )
    relative = path.relative_to(ROOT).as_posix()
    external_database_path = any(
        fragment in relative for fragment in EXTERNAL_DATABASE_SCHEMA_PATHS
    )
    literals: list[tuple[int, str]] = []
    if path.suffix == ".sql":
        literals.append((1, source))
    else:
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return violations
        literals.extend(
            (node.lineno, node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    for line, value in literals:
        normalized = " ".join(value.upper().split())
        if SELF_SCHEMA_PROBE.search(normalized):
            violations.append(
                f"{path}:{line}: 运行时禁止通过业务表探测自身 Schema"
            )
        if SELF_SCHEMA_VERSION_VIEW.search(normalized):
            violations.append(
                f"{path}:{line}: 运行时禁止读取自身 Schema 版本视图"
            )
        if external_database_path:
            continue
        for marker in DATABASE_CATALOG_MARKERS:
            if marker in normalized:
                violations.append(
                    f"{path}:{line}: 运行时禁止读取数据库系统目录 {marker}；"
                    "外部目标库发现必须放在已声明的适配器目录"
                )
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
    for root in RUNTIME_ROOTS:
        if not root.exists():
            continue
        for pattern in ("*.py", "*.sql"):
            for path in root.rglob(pattern):
                if (
                    "__pycache__" not in path.parts
                    and "build" not in path.parts
                ):
                    violations.extend(check_runtime_database_boundary(path))
    if violations:
        print("发现 KBot 4.0 架构边界违规：")
        print("\n".join(violations))
        return 1
    print("KBot 4.0 架构边界检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
