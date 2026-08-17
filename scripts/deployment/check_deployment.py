"""输出当前KBot部署配置摘要，并校验生产Secret。"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import tomli


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_PATH = ROOT / "resources" / "topology.toml"

from main_api.config import MainApiSettings
from platform_core.config import load_settings


def check_deployment(config_file: Path | None = None) -> list[str]:
    path = Path(
        config_file
        or os.getenv("KBOT_CONFIG_FILE")
        or ROOT / "configuration" / "kbot.toml"
    ).resolve()
    errors: list[str] = []
    try:
        settings = load_settings(
            MainApiSettings,
            service="main_api",
            config_file=path,
        )
    except Exception as exc:
        return [f"配置加载失败：{exc}"]

    try:
        with path.open("rb") as stream:
            raw = tomli.load(stream)
    except (OSError, tomli.TOMLDecodeError) as exc:
        return [f"部署配置无法读取：{exc}"]

    main_api_base_url = str(
        (raw.get("ui") or {}).get("main_api_base_url") or ""
    ).strip().rstrip("/")
    parsed_main_api = urlsplit(main_api_base_url)
    if (
        parsed_main_api.scheme not in {"http", "https"}
        or not parsed_main_api.netloc
        or parsed_main_api.username
        or parsed_main_api.password
        or parsed_main_api.path not in {"", "/"}
        or parsed_main_api.query
        or parsed_main_api.fragment
    ):
        errors.append("必须配置合法的[ui].main_api_base_url")
    elif settings.is_production() and (
        parsed_main_api.hostname or ""
    ).endswith("example.com"):
        errors.append("[ui].main_api_base_url仍使用示例地址")

    if settings.is_production():
        if not os.getenv("KBOT_ORACLE_PASSWORD"):
            errors.append("未设置KBOT_ORACLE_PASSWORD")
        master_key = os.getenv("KBOT_MASTER_KEY", "")
        if len(master_key.encode("utf-8")) < 32:
            errors.append("KBOT_MASTER_KEY必须至少32字节")
        if any(
            item.key_digest == "0" * 64
            for item in settings.security.portal_api_keys
        ):
            errors.append("Portal API Key 仍使用模板摘要，请生成后替换")
    return errors


def main() -> int:
    path = Path(
        os.getenv("KBOT_CONFIG_FILE")
        or ROOT / "configuration" / "kbot.toml"
    ).resolve()
    errors = check_deployment(path)
    if errors:
        print("KBot部署配置检查失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    with path.open("rb") as stream:
        raw = tomli.load(stream)
    with TOPOLOGY_PATH.open("rb") as stream:
        process_count = len(tomli.load(stream).get("processes") or ())
    settings = load_settings(
        MainApiSettings,
        service="main_api",
        config_file=path,
    )
    database = settings.database.oracle
    endpoint_count = len(raw.get("endpoints") or {})
    topology_mode = "跨主机覆盖" if endpoint_count else "本机自动拓扑"
    print("KBot部署配置检查通过")
    print(f"- 环境：{settings.environment}")
    print(
        f"- 数据库：{database.username}@{database.host}:"
        f"{database.port}/{database.service_name}"
    )
    print(f"- 数据目录：{raw.get('data_dir', './data')}")
    print(f"- 日志目录：{settings.log.dir}")
    print(f"- 文本向量维度：{settings.vector.dimensions}")
    print(f"- 内部服务：{process_count}个进程，{topology_mode}")
    if settings.is_production():
        print("- Secret：生产必需项已配置")
    else:
        print("- Secret：开发环境不强制校验")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
