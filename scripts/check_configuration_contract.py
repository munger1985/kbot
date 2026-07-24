"""校验 4.0 配置文件、Example、配置模型与 Secret 模板契约。"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.config import AgentRuntimeSettings  # noqa: E402
from aiops_agent.config import AIOpsSettings  # noqa: E402
from knowledge_core.config import KnowledgeCoreSettings  # noqa: E402
from main_api.config import MainApiSettings  # noqa: E402
from model_serving.config import ModelServingSettings  # noqa: E402
from platform_core.config import Settings  # noqa: E402


CONFIG_ROOT = ROOT / "configuration"
EXAMPLE_ROOT = CONFIG_ROOT / "example"
SERVICE_MODELS = {
    "agent_runtime": AgentRuntimeSettings,
    "aiops_agent": AIOpsSettings,
    "knowledge_core": KnowledgeCoreSettings,
    "main_api": MainApiSettings,
    "model_serving": ModelServingSettings,
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _source_path(source_root: Path, relative: str) -> Path:
    path = source_root / relative
    if source_root == EXAMPLE_ROOT:
        path = path.with_name(f"{path.name}.example")
    return path


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in update.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _leaf_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            paths.update(_leaf_paths(item, path))
        elif isinstance(item, list) and item and isinstance(item[0], dict):
            paths.add(f"{path}[]")
            paths.update(_leaf_paths(item[0], f"{path}[]"))
        else:
            paths.add(path)
    return paths


def _config_pairs() -> tuple[tuple[Path, Path], ...]:
    actual = (
        CONFIG_ROOT / "base.toml",
        CONFIG_ROOT / "development.toml",
        CONFIG_ROOT / "production.toml",
        *sorted((CONFIG_ROOT / "services").glob("*/*.toml")),
    )
    return tuple(
        (
            path,
            EXAMPLE_ROOT
            / Path(f"{path.relative_to(CONFIG_ROOT)}.example"),
        )
        for path in actual
    )


def _validate_model(
    model: type[Settings],
    source_root: Path,
    service: str | None,
    environment: str,
) -> None:
    merged = _merge(
        _load(_source_path(source_root, "base.toml")),
        _load(_source_path(source_root, f"{environment}.toml")),
    )
    if service:
        relative_root = f"services/{service}"
        merged = _merge(
            merged,
            _load(_source_path(source_root, f"{relative_root}/base.toml")),
        )
        merged = _merge(
            merged,
            _load(
                _source_path(
                    source_root,
                    f"{relative_root}/{environment}.toml",
                )
            ),
        )
    merged["environment"] = environment
    merged["config_dir"] = str(source_root)
    model.model_validate(merged)


def _declared_secret_env_names() -> set[str]:
    names = {"KBOT_AUTH_ENCRYPTION_KEY"}
    for source_root in (CONFIG_ROOT, EXAMPLE_ROOT):
        settings = AIOpsSettings.model_validate(
            {
                **_merge(
                    _merge(
                        _load(_source_path(source_root, "base.toml")),
                        _load(
                            _source_path(
                                source_root, "development.toml"
                            )
                        ),
                    ),
                    _load(
                        _source_path(
                            source_root,
                            "services/aiops_agent/base.toml",
                        )
                    ),
                ),
                "environment": "development",
                "config_dir": str(source_root),
            }
        )
        names.update(
            {
                settings.database.oracle.password_env,
                settings.security.api_key_pepper_env,
                settings.security.internal_service_token_env,
                settings.security.internal_jwt_secret_env,
                settings.security.service_identity_jwt_secret_env,
                settings.executor.grant_secret_env,
                settings.executor.mutation_grant_secret_env,
                settings.management.cursor_secret_env,
                settings.management.webhook_key_secret_env,
            }
        )
    return names


def check_configuration_contract() -> list[str]:
    errors: list[str] = []
    for actual, example in _config_pairs():
        if not example.is_file():
            errors.append(f"缺少配置 Example：{example.relative_to(ROOT)}")
            continue
        actual_paths = _leaf_paths(_load(actual))
        example_paths = _leaf_paths(_load(example))
        if actual_paths != example_paths:
            missing = sorted(actual_paths - example_paths)
            extra = sorted(example_paths - actual_paths)
            errors.append(
                f"{actual.relative_to(ROOT)} 与 Example 字段不一致："
                f"缺少={missing} 多余={extra}"
            )

    for source_root in (CONFIG_ROOT, EXAMPLE_ROOT):
        source_name = source_root.relative_to(ROOT)
        for environment in ("development", "production"):
            try:
                _validate_model(Settings, source_root, None, environment)
            except Exception as exc:
                errors.append(
                    f"{source_name} 共享 {environment} 配置无效：{exc}"
                )
            for service, model in SERVICE_MODELS.items():
                try:
                    _validate_model(
                        model, source_root, service, environment
                    )
                except Exception as exc:
                    errors.append(
                        f"{source_name}/{service} {environment} "
                        f"配置无效：{exc}"
                    )

    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    declared = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", env_text, flags=re.MULTILINE)
    )
    missing_secrets = sorted(_declared_secret_env_names() - declared)
    if missing_secrets:
        errors.append(f".env.example 缺少 Secret 变量：{missing_secrets}")
    return errors


def main() -> int:
    errors = check_configuration_contract()
    if errors:
        print("KBot 配置契约校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "KBot 配置契约校验通过："
        f"{len(_config_pairs())} 组配置/Example，"
        f"{len(SERVICE_MODELS)} 个服务配置模型"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
