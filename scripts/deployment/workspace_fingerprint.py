"""KBot 内部包源码与安装内容指纹。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class WorkspacePackage:
    """一个可独立安装的 KBot 内部包。"""

    member: str
    distribution: str
    module: str
    resource_patterns: tuple[str, ...] = ()

    @property
    def source_root(self) -> Path:
        return ROOT / self.member / "src" / self.module


WORKSPACE_PACKAGES = (
    WorkspacePackage(
        "packages/platform_core",
        "kbot-platform-core",
        "platform_core",
        ("resources/*.toml",),
    ),
    WorkspacePackage("packages/platform_clients", "kbot-platform-clients", "platform_clients"),
    WorkspacePackage("services/model_serving", "kbot-model-serving", "model_serving"),
    WorkspacePackage("services/knowledge_core", "kbot-knowledge-core", "knowledge_core"),
    WorkspacePackage(
        "services/knowledge_retrieval_app",
        "kbot-knowledge-retrieval-app",
        "knowledge_retrieval_app",
    ),
    WorkspacePackage("services/km_asset_app", "kbot-km-asset-app", "km_asset_app"),
    WorkspacePackage("services/agent_runtime", "kbot-agent-runtime", "agent_runtime"),
    WorkspacePackage(
        "services/aiops_agent",
        "kbot-aiops-agent",
        "aiops_agent",
        (
            "actions/catalog/**/*.json",
            "actions/catalog/**/*.sql",
            "diagnostics/catalog/**/*.json",
            "diagnostics/catalog/**/*.sql",
            "orchestration/diagnosis/prompt_assets/*.txt",
            "resources/**/*.json",
        ),
    ),
    WorkspacePackage("services/data_query", "kbot-data-query", "data_query"),
    WorkspacePackage("services/main_api", "kbot-main-api", "main_api"),
)


def _files(root: Path, resource_patterns: tuple[str, ...] = ()):
    """返回参与运行时指纹的源码和包内资源。"""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        if path.suffix != ".py" and not any(
            relative.match(pattern) for pattern in resource_patterns
        ):
            continue
        yield relative, path


def tree_fingerprint(
    root: Path, resource_patterns: tuple[str, ...] = ()
) -> tuple[str, tuple[Path, ...]]:
    """计算目录中全部运行时文件的稳定内容指纹。"""
    digest = sha256()
    files: list[Path] = []
    for relative, path in _files(root, resource_patterns):
        files.append(relative)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), tuple(files)


def installed_module_root(module: str) -> Path | None:
    """解析当前解释器实际加载的顶级包目录。"""
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def compare_workspace_packages() -> list[str]:
    """返回源码与当前解释器安装内容不一致的内部包。"""
    mismatches: list[str] = []
    for package in WORKSPACE_PACKAGES:
        source_root = package.source_root.resolve()
        installed_root = installed_module_root(package.module)
        if installed_root is None:
            mismatches.append(f"{package.distribution}:NOT_INSTALLED")
            continue
        if installed_root == source_root:
            continue
        source_fingerprint, source_files = tree_fingerprint(
            source_root, package.resource_patterns
        )
        missing = tuple(
            relative for relative in source_files
            if not (installed_root / relative).is_file()
        )
        if missing:
            mismatches.append(
                f"{package.distribution}:MISSING:{missing[0].as_posix()}"
            )
            continue
        installed_fingerprint = sha256()
        for relative in source_files:
            installed_fingerprint.update(relative.as_posix().encode("utf-8"))
            installed_fingerprint.update(b"\0")
            installed_fingerprint.update((installed_root / relative).read_bytes())
            installed_fingerprint.update(b"\0")
        if installed_fingerprint.hexdigest() != source_fingerprint:
            mismatches.append(f"{package.distribution}:CONTENT_MISMATCH")
    return mismatches


def main() -> int:
    """在当前全新解释器中输出内部包内容检查结果。"""
    mismatches = compare_workspace_packages()
    if mismatches:
        print("KBot 内部包内容指纹不一致：")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print("KBot 内部包内容指纹一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
