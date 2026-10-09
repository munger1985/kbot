#!/usr/bin/env python3
"""KBot 4.0 容器镜像与单机 Compose 交付渲染工具。"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
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


def deployment_values(path: Path) -> dict[str, Any]:
    parser = load_ini(path)
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
    oracle_password_file = resolved_path(
        required(parser, "secrets", "oracle_password_file"), relative_to=base
    )
    master_key_file = resolved_path(
        required(parser, "secrets", "master_key_file"), relative_to=base
    )
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
    lines = [f"name: {yaml_string(values['project_name'])}", "services:"]
    for process in topology["processes"]:
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
            "    ports:",
            f"      - {yaml_string(str(values['ui_port']) + ':8080')}",
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


def render_deployment(config_path: Path) -> Path:
    catalog, topology = validate_catalog()
    values = deployment_values(config_path)
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
    return output


def validate_command(
    release_config: Path | None, deployment_config: Path | None
) -> None:
    validate_catalog()
    if release_config:
        release_values(release_config)
    if deployment_config:
        deployment_values(deployment_config)


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
    secret = commands.add_parser("generate-master-key")
    secret.add_argument("--output", type=Path, required=True)
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
        elif arguments.command == "generate-master-key":
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(
                secrets.token_urlsafe(48) + "\n", encoding="utf-8"
            )
            arguments.output.chmod(0o600)
            print(arguments.output)
    except (ReleaseError, ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"发布工具执行失败：{error}") from error


if __name__ == "__main__":
    main()
