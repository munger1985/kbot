"""验收 KBot workspace 成员、发行包和模块来源。"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import re

import tomli


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = (
    ("packages/platform_core", "kbot-platform-core", "platform_core"),
    ("packages/platform_clients", "kbot-platform-clients", "platform_clients"),
    ("services/model_serving", "kbot-model-serving", "model_serving"),
    ("services/knowledge_core", "kbot-knowledge-core", "knowledge_core"),
    (
        "services/knowledge_retrieval_app",
        "kbot-knowledge-retrieval-app",
        "knowledge_retrieval_app",
    ),
    ("services/km_asset_app", "kbot-km-asset-app", "km_asset_app"),
    ("services/agent_runtime", "kbot-agent-runtime", "agent_runtime"),
    ("services/aiops_agent", "kbot-aiops-agent", "aiops_agent"),
    ("services/data_query", "kbot-data-query", "data_query"),
    ("services/main_api", "kbot-main-api", "main_api"),
)

AIOPS_RUNTIME_RESOURCE_SUFFIXES = frozenset({".json", ".sql", ".txt"})


def _resolved(path: Path) -> Path:
    return path.resolve(strict=False)


def check_workspace_packages() -> list[str]:
    """返回 workspace 声明、安装元数据和模块来源错误。"""
    errors: list[str] = []
    root_config = tomli.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    members = tuple(
        root_config.get("tool", {})
        .get("kbot", {})
        .get("workspace", {})
        .get("members", ())
    )
    expected_members = tuple(item[0] for item in EXPECTED)
    if members != expected_members:
        errors.append(
            f"workspace members 顺序不一致：actual={members!r} "
            f"expected={expected_members!r}"
        )

    mode = os.getenv("KBOT_WORKSPACE_PACKAGE_MODE", "development")
    if mode not in {"development", "production"}:
        errors.append(f"未知包验收模式：{mode}")
        return errors

    locked_dependencies = {
        match.group(1).lower().replace("_", "-"): match.group(2)
        for line in (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if (match := re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line))
    }

    for member, distribution_name, module_name in EXPECTED:
        pyproject_path = ROOT / member / "pyproject.toml"
        if not pyproject_path.is_file():
            errors.append(f"缺少工作区成员：{member}")
            continue
        config = tomli.loads(pyproject_path.read_text(encoding="utf-8"))
        project = config.get("project", {})
        if project.get("name") != distribution_name:
            errors.append(
                f"{member} 发行名错误：{project.get('name')!r}"
            )
        if project.get("version") != "4.0.0":
            errors.append(
                f"{member} 版本错误：{project.get('version')!r}"
            )
        for dependency in project.get("dependencies", ()):
            match = re.fullmatch(
                r"([A-Za-z0-9_.-]+)==([^\s]+)", dependency
            )
            if match is None:
                continue
            name = match.group(1).lower().replace("_", "-")
            locked = locked_dependencies.get(name)
            if locked is not None and locked != match.group(2):
                errors.append(
                    f"{member} 依赖未与 requirements.txt 对齐："
                    f"{dependency}，锁定版本={locked}"
                )
        includes = (
            config.get("tool", {})
            .get("setuptools", {})
            .get("packages", {})
            .get("find", {})
            .get("include", ())
        )
        if f"{module_name}*" not in includes:
            errors.append(f"{member} 未限定顶级包 {module_name}")

        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"内部发行包尚未安装：{distribution_name}")
            continue
        if distribution.version != "4.0.0":
            errors.append(
                f"{distribution_name} 安装版本错误：{distribution.version}"
            )

        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            errors.append(f"无法解析模块：{module_name}")
            continue
        origin = _resolved(Path(spec.origin))
        source_root = _resolved(ROOT / member / "src")
        from_source = origin.is_relative_to(source_root)
        if mode == "development" and not from_source:
            errors.append(
                f"开发模式模块未指向 editable 源码：{module_name}={origin}"
            )
        if mode == "production" and from_source:
            errors.append(
                f"生产模式模块仍指向仓库源码：{module_name}={origin}"
            )
        if module_name == "aiops_agent":
            source_package_root = source_root / "aiops_agent"
            runtime_resources = tuple(
                path.relative_to(source_package_root)
                for path in source_package_root.rglob("*")
                if path.is_file()
                and path.suffix in AIOPS_RUNTIME_RESOURCE_SUFFIXES
            )
            missing_resources = tuple(
                str(relative_path)
                for relative_path in runtime_resources
                if not (origin.parent / relative_path).is_file()
            )
            if missing_resources:
                errors.append(
                    "AIOps 发行包缺少运行资源："
                    + ", ".join(missing_resources)
                )
    return errors


def main() -> int:
    errors = check_workspace_packages()
    if errors:
        print("KBot workspace package 验收失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"KBot workspace package 验收通过：{len(EXPECTED)} 个内部包")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
