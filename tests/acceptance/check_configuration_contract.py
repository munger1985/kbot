"""验收单一部署配置、全部服务模型与 Secret 模板。"""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_runtime.config import AgentRuntimeSettings  # noqa: E402
from aiops_agent.config import AIOpsSettings  # noqa: E402
from knowledge_core.config import KnowledgeCoreSettings  # noqa: E402
from main_api.config import MainApiSettings  # noqa: E402
from model_serving.config import ModelServingSettings  # noqa: E402
from platform_core.config import Settings, load_settings  # noqa: E402


CONFIG_ROOT = ROOT / "configuration"
SERVICE_MODELS = {
    "agent_runtime": AgentRuntimeSettings,
    "aiops_agent": AIOpsSettings,
    "knowledge_core": KnowledgeCoreSettings,
    "main_api": MainApiSettings,
    "model_serving": ModelServingSettings,
}


def _required_config_files() -> tuple[Path, ...]:
    return (
        CONFIG_ROOT / "kbot.toml.example",
        ROOT / "resources" / "topology.toml",
    )


def _validate_config(path: Path) -> list[str]:
    errors: list[str] = []
    label = path.name
    try:
        load_settings(Settings, config_file=path)
    except Exception as exc:
        errors.append(f"{label} 平台配置无效：{exc}")
    for service, model in SERVICE_MODELS.items():
        try:
            load_settings(model, service=service, config_file=path)
        except Exception as exc:
            errors.append(f"{label} 的 {service} 配置无效：{exc}")
    return errors


def check_configuration_contract() -> list[str]:
    errors: list[str] = []
    for path in _required_config_files():
        if not path.is_file():
            errors.append(f"缺少配置文件：{path.relative_to(ROOT)}")
    if errors:
        return errors

    errors.extend(_validate_config(CONFIG_ROOT / "kbot.toml.example"))
    local_config = CONFIG_ROOT / "kbot.toml"
    if local_config.is_file():
        errors.extend(_validate_config(local_config))

    env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
    declared = set(
        re.findall(r"^([A-Z][A-Z0-9_]*)=", env_text, flags=re.MULTILINE)
    )
    required_secrets = {"KBOT_ORACLE_PASSWORD", "KBOT_MASTER_KEY"}
    missing = sorted(required_secrets - declared)
    if missing:
        errors.append(f".env.example 缺少必需 Secret：{missing}")
    return errors


def main() -> int:
    errors = check_configuration_contract()
    if errors:
        print("KBot 配置契约校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "KBot 配置契约校验通过：1 个部署模板，"
        f"{len(SERVICE_MODELS)} 个服务配置模型"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
