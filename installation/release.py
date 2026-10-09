#!/usr/bin/env python3
"""KBot 4.0 容器镜像与单机 Compose 交付渲染工具。"""

from __future__ import annotations

import argparse
import configparser
import getpass
import importlib.util
import io
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
INSTALLATION_ROOT = ROOT / "installation"
CATALOG_PATH = INSTALLATION_ROOT / "catalog.toml"
TOPOLOGY_PATH = ROOT / "resources" / "topology.toml"
DEFAULT_RELEASE_CONFIG = INSTALLATION_ROOT / "dev" / "release.ini"
DEFAULT_DEPLOYMENT_CONFIG = INSTALLATION_ROOT / "dev" / "deployment.ini"
VERSION_PATTERN = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,63}$")


class ReleaseError(RuntimeError):
    """可直接展示给发布人员的配置错误。"""


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def load_ini(path: Path) -> configparser.ConfigParser:
    if not path.is_file():
        raise ReleaseError(f"配置文件不存在：{path}")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    return parser


def required(
    parser: configparser.ConfigParser, section: str, option: str
) -> str:
    value = parser.get(section, option, fallback="").strip()
    if not value:
        raise ReleaseError(f"配置 [{section}] {option} 未填写")
    return value


def optional(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: str = "",
) -> str:
    return parser.get(section, option, fallback=fallback).strip()


def resolved_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def deployment_secret_paths(path: Path) -> tuple[Path, Path]:
    """从部署配置中解析 Oracle 密码与平台主密钥文件位置。"""

    if path.stat().st_mode & 0o077:
        raise ReleaseError(f"部署配置权限必须为 0600：chmod 600 {path}")
    parser = load_ini(path)
    base = path.parent
    oracle_password_file = resolved_path(
        required(parser, "secrets", "oracle_password_file"), relative_to=base
    )
    master_key_file = resolved_path(
        required(parser, "secrets", "master_key_file"), relative_to=base
    )
    if oracle_password_file == master_key_file:
        raise ReleaseError("Oracle 密码与 KBot 主密钥不能使用同一个 Secret 文件")
    return oracle_password_file, master_key_file


def write_private_secret(path: Path, value: str, *, replace: bool = False) -> None:
    """以原子替换方式写入仅当前用户可读写的 Secret 文件。"""

    if path.is_symlink():
        raise ReleaseError(f"Secret 文件不能是符号链接：{path}")
    if path.exists() and not replace:
        raise ReleaseError(f"Secret 文件已存在，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            target.write(value + "\n")
            target.flush()
            os.fsync(target.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def prepare_secrets(
    config_path: Path, *, replace_oracle_password: bool = False
) -> tuple[Path, Path]:
    """交互录入 Oracle 密码，并首次生成稳定的平台主密钥。"""

    oracle_password_file, master_key_file = deployment_secret_paths(config_path)
    if oracle_password_file.is_symlink() or (
        oracle_password_file.exists() and not oracle_password_file.is_file()
    ):
        raise ReleaseError(
            f"Oracle 密码 Secret 必须是普通文件：{oracle_password_file}"
        )
    if master_key_file.is_symlink() or (
        master_key_file.exists() and not master_key_file.is_file()
    ):
        raise ReleaseError(
            f"KBot 主密钥 Secret 必须是普通文件：{master_key_file}"
        )
    if master_key_file.exists():
        master_key = master_key_file.read_text(encoding="utf-8").strip()
        if len(master_key.encode("utf-8")) < 32:
            raise ReleaseError("已有 KBot 主密钥 Secret 少于 32 字节，拒绝覆盖")

    if oracle_password_file.exists() and not replace_oracle_password:
        if not oracle_password_file.read_text(encoding="utf-8").strip():
            raise ReleaseError("已有 Oracle 密码 Secret 为空，拒绝继续")
        oracle_password_file.chmod(0o600)
        print(f"保留已有 Oracle 密码 Secret：{oracle_password_file}")
    else:
        password = getpass.getpass("请输入 Oracle Schema 密码：")
        confirmation = getpass.getpass("请再次输入 Oracle Schema 密码：")
        if not password or any(character in password for character in "\r\n\0"):
            raise ReleaseError("Oracle 密码不能为空或包含换行符、空字符")
        if password != confirmation:
            raise ReleaseError("两次输入的 Oracle 密码不一致")
        write_private_secret(
            oracle_password_file,
            password,
            replace=replace_oracle_password,
        )
        print(f"Oracle 密码 Secret 已写入：{oracle_password_file}")

    if master_key_file.exists():
        master_key_file.chmod(0o600)
        print(f"保留已有 KBot 主密钥 Secret：{master_key_file}")
    else:
        write_private_secret(master_key_file, secrets.token_urlsafe(48))
        print(f"KBot 主密钥 Secret 已生成：{master_key_file}")
    return oracle_password_file, master_key_file


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return f"{revision}-dirty" if status.stdout.strip() else revision


def validate_catalog() -> tuple[dict[str, Any], dict[str, Any]]:
    catalog = load_toml(CATALOG_PATH)
    topology = load_toml(TOPOLOGY_PATH)
    images = catalog.get("images") or []
    image_ids = [str(item.get("id") or "") for item in images]
    if not image_ids or "" in image_ids or len(image_ids) != len(set(image_ids)):
        raise ReleaseError("catalog.toml 的镜像 id 为空或重复")
    image_names = [str(item.get("name") or "") for item in images]
    if "" in image_names or len(image_names) != len(set(image_names)):
        raise ReleaseError("catalog.toml 的镜像名称为空或重复")
    image_by_id = {str(item["id"]): item for item in images}
    service_configs = set(image_by_id)
    base_configs = set((catalog.get("base_pack") or {}).get("service_configs") or [])
    if not base_configs or not base_configs.issubset(service_configs):
        raise ReleaseError("catalog.toml 的 base_pack 引用了未知服务镜像")
    app_packs = catalog.get("app_packs") or {}
    if not app_packs:
        raise ReleaseError("catalog.toml 没有定义可部署 App")
    for app_id, pack in app_packs.items():
        selected = set(pack.get("service_configs") or [])
        if not selected or not selected.issubset(service_configs):
            raise ReleaseError(f"App {app_id} 引用了未知服务镜像")
        if not str(pack.get("entry_path") or "").startswith("/ui/"):
            raise ReleaseError(f"App {app_id} 缺少合法 UI 入口")

    processes = topology.get("processes") or []
    process_keys = [str(item.get("process_key") or "") for item in processes]
    if (
        not process_keys
        or "" in process_keys
        or len(process_keys) != len(set(process_keys))
    ):
        raise ReleaseError("topology.toml 的进程标识为空或重复")
    for process in processes:
        image_id = str(process.get("service_config") or "")
        if image_id not in image_by_id:
            raise ReleaseError(
                f"进程 {process['process_key']} 没有对应镜像：{image_id}"
            )
    process_by_key = {str(item["process_key"]): item for item in processes}
    for endpoint, process_key in (topology.get("endpoints") or {}).items():
        process = process_by_key.get(str(process_key))
        if not process or not process.get("port"):
            raise ReleaseError(f"端点 {endpoint} 引用了无 HTTP 端口的进程")
    return catalog, topology


def release_values(path: Path) -> dict[str, Any]:
    parser = load_ini(path)
    version = required(parser, "release", "version")
    if not VERSION_PATTERN.fullmatch(version):
        raise ReleaseError("[release] version 不是合法的镜像标签")
    registry = required(parser, "release", "registry").rstrip("/")
    if any(character.isspace() for character in registry):
        raise ReleaseError("[release] registry 不能包含空白字符")
    platform = required(parser, "release", "platform")
    if platform != "linux/amd64":
        raise ReleaseError("当前交付仅支持 linux/amd64")
    install_os_packages = optional(
        parser, "build", "install_os_packages", "true"
    ).lower()
    if install_os_packages not in {"true", "false"}:
        raise ReleaseError("[build] install_os_packages 只能是 true 或 false")
    python_runtime = {
        "torch_version": optional(
            parser, "python_runtime", "torch_version", "2.11.0+cpu"
        ),
        "torchvision_version": optional(
            parser, "python_runtime", "torchvision_version", "0.26.0+cpu"
        ),
        "torch_index_url": optional(
            parser,
            "python_runtime",
            "torch_index_url",
            "https://download.pytorch.org/whl/cpu",
        ),
    }
    if any(
        not value or any(character.isspace() for character in value)
        for value in python_runtime.values()
    ):
        raise ReleaseError("[python_runtime] CPU Torch 配置为空或包含空白字符")
    return {
        "version": version,
        "registry": registry,
        "platform": platform,
        "python_image": optional(
            parser, "base_images", "python", "python:3.12-slim-bookworm"
        ),
        "nginx_image": optional(
            parser, "base_images", "nginx", "nginx:1.27-alpine"
        ),
        "pip_index_url": optional(
            parser, "package_mirrors", "pip_index_url", "https://pypi.org/simple"
        ),
        "install_os_packages": install_os_packages,
        "python_runtime": python_runtime,
        "revision": git_revision(),
    }


def image_reference(values: dict[str, Any], name: str) -> str:
    return f"{values['registry']}/{name}:{values['version']}"


def build_directory() -> Path:
    return INSTALLATION_ROOT / "generated" / "build"


def deployment_directory() -> Path:
    return INSTALLATION_ROOT / "generated" / "deployment"


def hcl_map(values: dict[str, str]) -> str:
    lines = ["  args = {"]
    lines.extend(
        f"    {name} = {json.dumps(value)}" for name, value in values.items()
    )
    lines.append("  }")
    return "\n".join(lines)


def render_build(config_path: Path) -> Path:
    catalog, _ = validate_catalog()
    values = release_values(config_path)
    output = build_directory()
    output.mkdir(parents=True, exist_ok=True)
    targets: list[str] = []
    blocks: list[str] = []
    dockerfile = "installation/common/docker/Dockerfile.python"
    for image in catalog["images"]:
        target = str(image["id"])
        targets.append(target)
        packages = " ".join(str(item) for item in image["packages"])
        blocks.append(
            "\n".join(
                [
                    f'target "{target}" {{',
                    f"  context = {json.dumps(str(ROOT))}",
                    f"  dockerfile = {json.dumps(dockerfile)}",
                    "  tags = ["
                    + json.dumps(image_reference(values, str(image["name"])))
                    + "]",
                    f"  platforms = [{json.dumps(values['platform'])}]",
                    hcl_map(
                        {
                            "PYTHON_IMAGE": values["python_image"],
                            "PIP_INDEX_URL": values["pip_index_url"],
                            "INSTALL_OS_PACKAGES": values["install_os_packages"],
                            "INSTALL_CPU_TORCH": str(
                                bool(image.get("cpu_torch"))
                            ).lower(),
                            "TORCH_VERSION": values["python_runtime"]["torch_version"],
                            "TORCHVISION_VERSION": values["python_runtime"][
                                "torchvision_version"
                            ],
                            "TORCH_INDEX_URL": values["python_runtime"][
                                "torch_index_url"
                            ],
                            "SERVICE_PACKAGES": packages,
                            "VERSION": values["version"],
                            "REVISION": values["revision"],
                        }
                    ),
                    "}",
                ]
            )
        )
    installer = catalog["installer"]
    targets.append("installer")
    blocks.append(
        "\n".join(
            [
                'target "installer" {',
                f"  context = {json.dumps(str(ROOT))}",
                f"  dockerfile = {json.dumps(dockerfile)}",
                "  tags = ["
                + json.dumps(image_reference(values, str(installer["name"])))
                + "]",
                f"  platforms = [{json.dumps(values['platform'])}]",
                hcl_map(
                    {
                        "PYTHON_IMAGE": values["python_image"],
                        "PIP_INDEX_URL": values["pip_index_url"],
                        "INSTALL_OS_PACKAGES": values["install_os_packages"],
                        "INSTALL_CPU_TORCH": "true",
                        "TORCH_VERSION": values["python_runtime"]["torch_version"],
                        "TORCHVISION_VERSION": values["python_runtime"][
                            "torchvision_version"
                        ],
                        "TORCH_INDEX_URL": values["python_runtime"]["torch_index_url"],
                        "SERVICE_PACKAGES": " ".join(installer["packages"]),
                        "VERSION": values["version"],
                        "REVISION": values["revision"],
                    }
                ),
                "}",
            ]
        )
    )
    targets.append("ui")
    blocks.append(
        "\n".join(
            [
                'target "ui" {',
                f"  context = {json.dumps(str(ROOT))}",
                '  dockerfile = "installation/common/docker/Dockerfile.ui"',
                "  tags = ["
                + json.dumps(image_reference(values, str(catalog["ui"]["name"])))
                + "]",
                f"  platforms = [{json.dumps(values['platform'])}]",
                hcl_map(
                    {
                        "NGINX_IMAGE": values["nginx_image"],
                        "VERSION": values["version"],
                        "REVISION": values["revision"],
                    }
                ),
                "}",
            ]
        )
    )
    group = 'group "default" {\n  targets = [' + ", ".join(
        json.dumps(item) for item in targets
    ) + "]\n}"
    path = output / "docker-bake.hcl"
    path.write_text(group + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def build_images(config_path: Path, mode: str, target: str) -> None:
    if mode == "push" and git_revision().endswith("-dirty"):
        raise ReleaseError("工作区存在未提交修改，禁止推送不可追溯镜像")
    bake = render_build(config_path)
    command = ["docker", "buildx", "bake", "--file", str(bake)]
    if mode == "print":
        command.append("--print")
    elif mode == "load":
        command.append("--load")
    elif mode == "push":
        command.append("--push")
    if target != "default":
        command.append(target)
    subprocess.run(command, cwd=ROOT, check=True)


def deployment_values(
    path: Path, catalog: dict[str, Any] | None = None
) -> dict[str, Any]:
    if catalog is None:
        catalog, _ = validate_catalog()
    parser = load_ini(path)
    oracle_password_file, master_key_file = deployment_secret_paths(path)
    base = path.parent
    version = required(parser, "deployment", "image_version")
    if not VERSION_PATTERN.fullmatch(version):
        raise ReleaseError("[deployment] image_version 不是合法的镜像标签")
    registry = required(parser, "deployment", "registry").rstrip("/")
    if any(character.isspace() for character in registry):
        raise ReleaseError("[deployment] registry 不能包含空白字符")
    ui_port = parser.getint("deployment", "ui_port", fallback=8080)
    if not 1 <= ui_port <= 65535:
        raise ReleaseError("[deployment] ui_port 必须在 1 到 65535 之间")
    for label, secret_path in (
        ("Oracle 密码", oracle_password_file),
        ("KBot 主密钥", master_key_file),
    ):
        if not secret_path.is_file() or not secret_path.read_text(
            encoding="utf-8"
        ).strip():
            raise ReleaseError(f"{label} Secret 不存在或为空：{secret_path}")
    master_key = master_key_file.read_text(encoding="utf-8").strip()
    if len(master_key.encode("utf-8")) < 32:
        raise ReleaseError("KBot 主密钥 Secret 必须至少为 32 字节")
    data_dir = resolved_path(
        required(parser, "paths", "data_dir"), relative_to=base
    )
    log_dir = resolved_path(
        required(parser, "paths", "log_dir"), relative_to=base
    )
    public_base_url = required(
        parser, "deployment", "public_base_url"
    ).rstrip("/")
    parsed_public_url = urlparse(public_base_url)
    if (
        parsed_public_url.scheme not in {"http", "https"}
        or not parsed_public_url.netloc
    ):
        raise ReleaseError("[deployment] public_base_url 必须是完整的 HTTP(S) URL")
    database_port = parser.getint("database", "port", fallback=1521)
    if not 1 <= database_port <= 65535:
        raise ReleaseError("[database] port 必须在 1 到 65535 之间")
    embedding_dimension = parser.getint(
        "deployment", "embedding_dimension", fallback=2048
    )
    if embedding_dimension < 1:
        raise ReleaseError("[deployment] embedding_dimension 必须大于 0")
    project_name = required(parser, "deployment", "project_name")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", project_name):
        raise ReleaseError("[deployment] project_name 只能包含小写字母、数字、_ 和 -")
    known_apps = set(catalog["app_packs"])
    if not parser.has_section("apps"):
        raise ReleaseError("部署配置缺少 [apps] App 选择段")
    unknown_apps = sorted(set(parser["apps"]) - known_apps)
    if unknown_apps:
        raise ReleaseError(
            "[apps] 包含未知 App：" + ", ".join(unknown_apps)
        )
    try:
        enabled_apps = [
            app_id
            for app_id in catalog["app_packs"]
            if parser.getboolean("apps", app_id, fallback=False)
        ]
    except ValueError as error:
        raise ReleaseError("[apps] 的 App 选择必须是 true 或 false") from error
    if not enabled_apps:
        raise ReleaseError("[apps] 至少选择一个 App")
    observability_enabled = parser.getboolean(
        "observability", "enabled", fallback=False
    )
    if observability_enabled and "aiops" not in enabled_apps:
        raise ReleaseError("启用 AIOps 观测栈前必须在 [apps] 中选择 aiops")
    observability_sections = {
        section: dict(parser.items(section))
        for section in parser.sections()
        if section == "observability" or section.startswith("observability.")
    }
    return {
        "project_name": project_name,
        "version": version,
        "registry": registry,
        "ui_port": ui_port,
        "public_base_url": public_base_url,
        "database_host": required(parser, "database", "host"),
        "database_port": database_port,
        "database_service_name": required(parser, "database", "service_name"),
        "database_username": required(parser, "database", "username"),
        "embedding_dimension": embedding_dimension,
        "enabled_apps": enabled_apps,
        "observability_enabled": observability_enabled,
        "observability_sections": observability_sections,
        "aiops_agent_execution_enabled": parser.getboolean(
            "aiops", "agent_execution_enabled", fallback=False
        ),
        "aiops_mutation_enabled": parser.getboolean(
            "aiops", "mutation_enabled", fallback=False
        ),
        "oracle_password_file": oracle_password_file,
        "master_key_file": master_key_file,
        "data_dir": data_dir,
        "log_dir": log_dir,
    }


def toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def compose_name(process_key: str) -> str:
    return process_key.replace("_", "-")


def selected_service_configs(
    catalog: dict[str, Any], values: dict[str, Any]
) -> set[str]:
    """返回基础运行层与所选 App 的服务镜像并集。"""

    selected = set(catalog["base_pack"]["service_configs"])
    for app_id in values["enabled_apps"]:
        selected.update(catalog["app_packs"][app_id]["service_configs"])
    return selected


def render_schema_services(
    catalog: dict[str, Any], values: dict[str, Any]
) -> str:
    selected = selected_service_configs(catalog, values)
    lines = [
        "; 由 KBot 部署配置生成；platform_core 始终由初始化器自动加入。",
        "[services]",
    ]
    for image in catalog["images"]:
        service_config = str(image["id"])
        lines.append(
            f"{service_config} = {str(service_config in selected).lower()}"
        )
    return "\n".join(lines) + "\n"


def render_ui_runtime_config(values: dict[str, Any]) -> str:
    payload = {
        "mainApiBaseUrl": "",
        "enabledApps": values["enabled_apps"],
    }
    return (
        "globalThis.KBOT_UI_CONFIG = Object.freeze("
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ");\n"
    )


def render_runtime_toml(topology: dict[str, Any], values: dict[str, Any]) -> str:
    processes = {
        str(item["process_key"]): item for item in topology["processes"]
    }
    lines = [
        'environment = "production"',
        'data_dir = "/var/lib/kbot"',
        'log_dir = "/var/log/kbot"',
        f"embedding_dimension = {values['embedding_dimension']}",
        "api_docs_enabled = false",
        "",
        "[ui]",
        f"main_api_base_url = {toml_string(values['public_base_url'])}",
        "",
        "[database]",
        f"host = {toml_string(values['database_host'])}",
        f"port = {values['database_port']}",
        f"service_name = {toml_string(values['database_service_name'])}",
        f"username = {toml_string(values['database_username'])}",
        "",
        "[aiops]",
        "agent_execution_enabled = "
        + str(values["aiops_agent_execution_enabled"]).lower(),
        "mutation_enabled = " + str(values["aiops_mutation_enabled"]).lower(),
        "",
        "[endpoints]",
    ]
    for endpoint, process_key in topology["endpoints"].items():
        process = processes[str(process_key)]
        url = f"http://{compose_name(str(process_key))}:{process['port']}"
        lines.append(f"{endpoint} = {toml_string(url)}")
    return "\n".join(lines) + "\n"


def yaml_string(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_compose(
    catalog: dict[str, Any], topology: dict[str, Any], values: dict[str, Any]
) -> str:
    image_by_id = {str(item["id"]): item for item in catalog["images"]}
    selected_configs = selected_service_configs(catalog, values)
    lines = [f"name: {yaml_string(values['project_name'])}", "services:"]
    for process in topology["processes"]:
        if str(process["service_config"]) not in selected_configs:
            continue
        process_key = str(process["process_key"])
        service = compose_name(process_key)
        image = image_by_id[str(process["service_config"])]
        lines.extend(
            [
                f"  {service}:",
                "    image: "
                + yaml_string(image_reference(values, str(image["name"]))),
                "    restart: unless-stopped",
                "    init: true",
                f"    user: {yaml_string('${KBOT_UID}:${KBOT_GID}')}",
                "    command:",
                '      - "python"',
                '      - "-m"',
                f"      - {yaml_string(process['module'])}",
                "    environment:",
                '      KBOT_CONFIG_FILE: "/etc/kbot/kbot.toml"',
                '      KBOT_RESOURCE_DIR: "/opt/kbot/resources"',
                "    volumes:",
                '      - "./config/kbot.toml:/etc/kbot/kbot.toml:ro"',
                f"      - {yaml_string(str(values['data_dir']) + ':/var/lib/kbot')}",
                f"      - {yaml_string(str(values['log_dir']) + ':/var/log/kbot')}",
                "    secrets:",
                "      - kbot_oracle_password",
                "      - kbot_master_key",
                "    networks:",
                "      - kbot-internal",
            ]
        )
    lines.extend(
        [
            "  ui:",
            "    image: "
            + yaml_string(image_reference(values, str(catalog["ui"]["name"]))),
            "    restart: unless-stopped",
            "    environment:",
            '      MAIN_API_UPSTREAM: "main-api:18099"',
            "      UI_ENTRY_PATH: "
            + yaml_string(
                catalog["app_packs"][values["enabled_apps"][0]]["entry_path"]
            ),
            "    ports:",
            f"      - {yaml_string(str(values['ui_port']) + ':8080')}",
            "    volumes:",
            '      - "./config/runtime-config.js:'
            '/usr/share/nginx/html/ui/runtime-config.js:ro"',
            "    depends_on:",
            "      - main-api",
            "    networks:",
            "      - kbot-internal",
            "  installer:",
            "    image: "
            + yaml_string(
                image_reference(values, str(catalog["installer"]["name"]))
            ),
            '    profiles: ["tools"]',
            f"    user: {yaml_string('${KBOT_UID}:${KBOT_GID}')}",
            "    command:",
            '      - "python"',
            '      - "/opt/kbot/scripts/db/apply_oracle_schema.py"',
            '      - "--help"',
            "    environment:",
            '      KBOT_CONFIG_FILE: "/etc/kbot/kbot.toml"',
            '      KBOT_RESOURCE_DIR: "/opt/kbot/resources"',
            "    volumes:",
            '      - "./config/kbot.toml:/etc/kbot/kbot.toml:ro"',
            '      - "./config/oracle_schema_services.ini:'
            '/opt/kbot/configuration/oracle_schema_services.ini:ro"',
            f"      - {yaml_string(str(values['data_dir']) + ':/var/lib/kbot')}",
            f"      - {yaml_string(str(values['log_dir']) + ':/var/log/kbot')}",
            "    secrets:",
            "      - kbot_oracle_password",
            "      - kbot_master_key",
            "    networks:",
            "      - kbot-internal",
            "networks:",
            "  kbot-internal:",
            "    internal: false",
            "secrets:",
            "  kbot_oracle_password:",
            '    file: "./secrets/oracle_password"',
            "  kbot_master_key:",
            '    file: "./secrets/master_key"',
        ]
    )
    return "\n".join(lines) + "\n"


def private_copy(source: Path, destination: Path) -> None:
    value = source.read_text(encoding="utf-8").strip()
    destination.write_text(value + "\n", encoding="utf-8")
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)


def load_aiops_stack_module() -> Any:
    """加载无扩展名的既有 AIOps 观测栈发布器。"""

    module_name = "kbot_installation_aiops_stack"
    loader = SourceFileLoader(module_name, str(ROOT / "scripts" / "aiops-stack"))
    specification = importlib.util.spec_from_loader(module_name, loader)
    if specification is None or specification.loader is None:
        raise ReleaseError("无法加载 AIOps 观测栈发布器")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def observability_config_text(values: dict[str, Any]) -> str:
    """将统一配置单中的观测段转换为既有观测栈配置。"""

    sections = values["observability_sections"]
    root_values = sections.get("observability") or {}
    allowed_root = {
        "enabled",
        "deployment_id",
        "role",
        "local_access",
        "minimum_free_gb",
    }
    unknown_root = sorted(set(root_values) - allowed_root)
    if unknown_root:
        raise ReleaseError(
            "[observability] 包含未知字段：" + ", ".join(unknown_root)
        )
    parser = configparser.ConfigParser(interpolation=None)
    parser["deployment"] = {
        "deployment_id": root_values.get(
            "deployment_id", f"{values['project_name']}-observability"
        ),
        "role": root_values.get("role", "all-in-one"),
        "local_access": root_values.get("local_access", "false"),
        "minimum_free_gb": root_values.get("minimum_free_gb", "10"),
    }
    for source_section, section_values in sections.items():
        if not source_section.startswith("observability."):
            continue
        target_section = source_section.removeprefix("observability.")
        if not (
            target_section in {"metrics", "logs", "dashboard", "host"}
            or re.fullmatch(
                r"(?:oracle|mysql|postgres|prometheus_target):"
                r"[a-z0-9][a-z0-9_-]*",
                target_section,
            )
        ):
            raise ReleaseError(f"未知观测组件配置段：[{source_section}]")
        parser[target_section] = section_values
    buffer = io.StringIO()
    parser.write(buffer)
    return buffer.getvalue()


def validate_observability(values: dict[str, Any]) -> None:
    if not values["observability_enabled"]:
        return
    with tempfile.TemporaryDirectory(
        prefix="kbot-observability-validate-"
    ) as temporary:
        config_path = Path(temporary) / "aiops-stack.ini"
        config_path.write_text(observability_config_text(values), encoding="utf-8")
        config_path.chmod(0o600)
        module = load_aiops_stack_module()
        module._load_settings(config_path)


def render_observability(
    output: Path, values: dict[str, Any]
) -> dict[str, Any] | None:
    """把统一配置单映射给既有 AIOps 观测栈并执行只读渲染。"""

    if not values["observability_enabled"]:
        return None
    config_path = output / "observability" / "aiops-stack.ini"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(observability_config_text(values), encoding="utf-8")
    config_path.chmod(0o600)

    module = load_aiops_stack_module()
    settings = module._load_settings(config_path)
    module._prepare_runtime(settings)
    return {
        "role": settings.role,
        "profiles": list(settings.profiles),
        "services": module._selected_services(settings),
        "config_file": str(config_path),
    }


def render_deployment(config_path: Path) -> Path:
    catalog, topology = validate_catalog()
    values = deployment_values(config_path, catalog)
    validate_observability(values)
    output = deployment_directory()
    if output.exists():
        shutil.rmtree(output)
    (output / "config").mkdir(parents=True)
    (output / "secrets").mkdir()
    values["data_dir"].mkdir(parents=True, exist_ok=True)
    values["log_dir"].mkdir(parents=True, exist_ok=True)
    (output / "config" / "kbot.toml").write_text(
        render_runtime_toml(topology, values), encoding="utf-8"
    )
    (output / "config" / "kbot.toml").chmod(0o640)
    (output / "config" / "runtime-config.js").write_text(
        render_ui_runtime_config(values), encoding="utf-8"
    )
    (output / "config" / "oracle_schema_services.ini").write_text(
        render_schema_services(catalog, values), encoding="utf-8"
    )
    private_copy(
        values["oracle_password_file"], output / "secrets" / "oracle_password"
    )
    private_copy(values["master_key_file"], output / "secrets" / "master_key")
    (output / "compose.yaml").write_text(
        render_compose(catalog, topology, values), encoding="utf-8"
    )
    (output / ".env").write_text(
        f"KBOT_UID={os.getuid()}\nKBOT_GID={os.getgid()}\n", encoding="utf-8"
    )
    observability = render_observability(output, values)
    (output / "installation-input.json").write_text(
        json.dumps(
            {
                "schema": "kbot-installation-input.v1",
                "project_name": values["project_name"],
                "image_version": values["version"],
                "enabled_apps": values["enabled_apps"],
                "service_configs": sorted(
                    selected_service_configs(catalog, values)
                ),
                "observability": observability,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def deploy(config_path: Path) -> None:
    """先渲染并校验全部配置，再应用 KBot 与可选观测栈。"""

    output = render_deployment(config_path)
    compose = [
        "docker",
        "compose",
        "--project-directory",
        str(output),
        "-f",
        str(output / "compose.yaml"),
    ]
    observability_config = output / "observability" / "aiops-stack.ini"
    if observability_config.is_file():
        module = load_aiops_stack_module()
        settings = module._load_settings(observability_config)
        module._preflight(settings)
        subprocess.run(
            [*module._compose_command(settings), "config", "--quiet"],
            check=True,
        )
    subprocess.run([*compose, "config", "--quiet"], check=True)
    subprocess.run(
        [*compose, "up", "--detach", "--remove-orphans"], check=True
    )
    if observability_config.is_file():
        environment = os.environ.copy()
        environment["KBOT_AIOPS_STACK_CONFIG_FILE"] = str(
            observability_config
        )
        subprocess.run(
            [str(ROOT / "scripts" / "aiops-stack")],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print(f"KBot 部署已应用：{output}")


def validate_command(
    release_config: Path | None, deployment_config: Path | None
) -> None:
    validate_catalog()
    if release_config:
        release_values(release_config)
    if deployment_config:
        catalog, _ = validate_catalog()
        values = deployment_values(deployment_config, catalog)
        validate_observability(values)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--release-config", type=Path)
    validate.add_argument("--deployment-config", type=Path)
    render = commands.add_parser("render-build")
    render.add_argument("--config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    build = commands.add_parser("build")
    build.add_argument("--config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    build.add_argument("--mode", choices=("print", "load", "push"), default="load")
    build.add_argument("--target", default="default")
    deployment = commands.add_parser("render-deployment")
    deployment.add_argument(
        "--config", type=Path, default=DEFAULT_DEPLOYMENT_CONFIG
    )
    deploy_command = commands.add_parser("deploy")
    deploy_command.add_argument(
        "--config", type=Path, default=DEFAULT_DEPLOYMENT_CONFIG
    )
    secret = commands.add_parser("generate-master-key")
    secret.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-secrets",
        help="交互录入 Oracle 密码并首次生成 KBot 主密钥",
    )
    prepare.add_argument(
        "--config", type=Path, default=DEFAULT_DEPLOYMENT_CONFIG
    )
    prepare.add_argument(
        "--replace-oracle-password",
        action="store_true",
        help="明确更新已有 Oracle 密码 Secret；不会更换 KBot 主密钥",
    )
    return root


def main() -> None:
    arguments = parser().parse_args()
    try:
        if arguments.command == "validate":
            validate_command(arguments.release_config, arguments.deployment_config)
            print("KBot 容器交付配置校验通过")
        elif arguments.command == "render-build":
            print(render_build(arguments.config))
        elif arguments.command == "build":
            build_images(arguments.config, arguments.mode, arguments.target)
        elif arguments.command == "render-deployment":
            print(render_deployment(arguments.config))
        elif arguments.command == "deploy":
            deploy(arguments.config)
        elif arguments.command == "generate-master-key":
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                secrets.token_urlsafe(48) + "\n", encoding="utf-8"
            )
            arguments.output.chmod(0o600)
            print(arguments.output)
        elif arguments.command == "prepare-secrets":
            prepare_secrets(
                arguments.config,
                replace_oracle_password=arguments.replace_oracle_password,
            )
    except (ReleaseError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"发布工具执行失败：{error}") from error


if __name__ == "__main__":
    main()
