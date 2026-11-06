#!/usr/bin/env python3
"""以一份配置编排镜像、模型、空库初始化与部署验收。"""

from __future__ import annotations

import configparser
import fcntl
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if sys.version_info < (3, 11):
    raise SystemExit("部署入口需要 Python 3.11+，不需要安装 Python 第三方依赖。")
import release


STATE_SCHEMA = "kbot-installation-state.v1"


def private_json(path: Path, value: dict[str, Any]) -> None:
    release.write_private_secret(
        path, json.dumps(value, ensure_ascii=False, indent=2), replace=path.exists()
    )


def verify_package(root: Path) -> dict[str, Any]:
    """先验证发布工件，再解析镜像锁；本地配置与状态不在校验清单内。"""
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise release.ReleaseError("缺少正式部署包与 SHA256SUMS；请使用 export-deployment 或 export-offline 制作的交付包")
    checked: set[str] = set()
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise release.ReleaseError("SHA256SUMS 格式不正确")
        digest, relative = match.groups()
        path = root / relative
        if (
            Path(relative).is_absolute() or ".." in Path(relative).parts
            or path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
            or relative in checked or not path.is_file()
        ):
            raise release.ReleaseError("SHA256SUMS 包含重复、不安全或缺失的文件")
        if release.sha256_file(path) != digest:
            raise release.ReleaseError(f"部署工件校验失败：{relative}")
        checked.add(relative)
    required_files = {
        "bundle-manifest.json", "installation/release.py", "installation/deployment.py",
        "installation/catalog.toml", "resources/topology.toml", "kbot-deploy",
        "deployment.ini.example",
    }
    if not required_files.issubset(checked):
        raise release.ReleaseError("部署包校验清单没有覆盖全部必要工件")
    manifest = release.load_offline_bundle_manifest(root / "bundle-manifest.json")
    if manifest.get("platform") != "linux/amd64":
        raise release.ReleaseError("当前只支持 linux/amd64 部署包")
    if not release.VERSION_PATTERN.fullmatch(str(manifest.get("version", ""))):
        raise release.ReleaseError("部署包缺少合法版本")
    catalog, _ = release.validate_catalog()
    records = manifest.get("images", [])
    if [item.get("name") for item in records] != release.catalog_image_names(catalog):
        raise release.ReleaseError("镜像锁的范围、顺序或名称与服务目录不一致")
    for record in records:
        digest = record.get("digest", record.get("registry_digest", ""))
        if not release.IMAGE_ID_PATTERN.fullmatch(digest):
            raise release.ReleaseError("镜像锁包含非法 Digest")
        if manifest["schema"] == release.OFFLINE_BUNDLE_SCHEMA and not release.IMAGE_ID_PATTERN.fullmatch(record.get("image_id", "")):
            raise release.ReleaseError("离线镜像锁缺少合法 Image ID")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("image_source_commit", ""))):
        raise release.ReleaseError("部署包缺少完整源码修订")
    registry = str(manifest.get("registry", ""))
    if not registry or any(character.isspace() for character in registry):
        raise release.ReleaseError("部署包缺少合法镜像仓库地址")
    models = manifest.get("docling_models", {})
    if not {models.get("archive"), models.get("manifest")}.issubset(checked):
        raise release.ReleaseError("部署包校验清单没有覆盖模型资产")
    if manifest["schema"] == release.OFFLINE_BUNDLE_SCHEMA and manifest.get("image_archive") not in checked:
        raise release.ReleaseError("部署包校验清单没有覆盖离线镜像")
    return manifest


def load_input(path: Path, manifest: dict[str, Any]) -> tuple[configparser.ConfigParser, Path]:
    if path.is_symlink() or path.stat().st_mode & 0o077:
        raise release.ReleaseError(f"配置必须为普通私有文件：chmod 600 {path}")
    parser = release.load_ini(path)
    allowed = {
        "deployment": {"project_name", "public_base_url", "ui_port", "embedding_dimension"},
        "database": {"host", "port", "service_name", "username", "password"},
        "apps": {"knowledge_retrieval", "media_studio", "aiops"},
        "paths": {"data_dir", "log_dir"},
        "installation": {"initialize_empty_schema", "adopt_existing_schema", "timeout_seconds"},
        "aiops": {"agent_execution_enabled", "mutation_enabled"},
    }
    if parser.defaults():
        raise release.ReleaseError("配置不支持 DEFAULT 段，请在对应段显式填写")
    for section in parser.sections():
        if section == "observability" or section.startswith("observability."):
            continue
        if section not in allowed or set(parser[section]) - allowed[section]:
            raise release.ReleaseError(f"配置包含未知段或字段：[{section}]")
    password = release.required(parser, "database", "password")
    if password == "CHANGE_ME" or any(character in password for character in "\r\n\0"):
        raise release.ReleaseError("请填写有效的 Oracle 密码，不能包含换行或空字符")
    for key in ("host", "service_name", "username"):
        release.required(parser, "database", key)
    public_url = urlparse(release.required(parser, "deployment", "public_base_url"))
    if public_url.username or public_url.password or public_url.query or public_url.fragment:
        raise release.ReleaseError("公开访问地址不能包含凭据、查询参数或片段")
    data_dir = release.resolved_path(release.required(parser, "paths", "data_dir"), relative_to=path.parent)
    log_dir = release.resolved_path(release.required(parser, "paths", "log_dir"), relative_to=path.parent)
    if data_dir == log_dir or data_dir == Path("/") or log_dir == Path("/"):
        raise release.ReleaseError("数据目录与日志目录必须独立，且不能是根目录")
    parser["paths"]["data_dir"] = str(data_dir)
    parser["paths"]["log_dir"] = str(log_dir)
    parser["deployment"]["registry"] = manifest["registry"]
    parser["deployment"]["image_version"] = manifest["version"]
    timeout = parser.getint("installation", "timeout_seconds", fallback=300)
    if not 30 <= timeout <= 3600:
        raise release.ReleaseError("timeout_seconds 必须在 30 到 3600 之间")
    if parser.getboolean("installation", "initialize_empty_schema", fallback=False) and parser.getboolean("installation", "adopt_existing_schema", fallback=False):
        raise release.ReleaseError("空库初始化与接管已有部署不能同时启用")
    return parser, data_dir / ".kbot"


@contextmanager
def installation_lock(state_dir: Path):
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if state_dir.is_symlink():
        raise release.ReleaseError("安装状态目录不能是符号链接")
    state_dir.chmod(0o700)
    descriptor = os.open(state_dir / "deployment.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise release.ReleaseError("同一数据目录已有部署任务正在执行") from error
        yield
    finally:
        os.close(descriptor)


def read_state(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "installation.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise release.ReleaseError("安装记录不能是符号链接")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != STATE_SCHEMA:
        raise release.ReleaseError("安装记录格式不正确，不能自动接管")
    return state


def prepare_config(parser: configparser.ConfigParser, state_dir: Path, state: dict[str, Any] | None) -> tuple[Path, dict[str, Any]]:
    """将唯一配置拆为运行时配置和 Secret；主密钥永远不自动轮换。"""
    master = state_dir / "master_key"
    adopting = parser.getboolean("installation", "adopt_existing_schema", fallback=False)
    if master.is_symlink():
        raise release.ReleaseError("主密钥不能是符号链接")
    if not master.exists():
        if state is not None or adopting:
            raise release.ReleaseError("已有部署缺少原主密钥；请从受控备份恢复，禁止重新生成")
        release.write_private_secret(master, release.secrets.token_urlsafe(48))
    if len(master.read_text(encoding="utf-8").strip().encode("utf-8")) < 32:
        raise release.ReleaseError("主密钥少于 32 字节，拒绝继续")
    master.chmod(0o600)
    password_file = state_dir / "oracle_password"
    release.write_private_secret(password_file, parser["database"]["password"], replace=password_file.exists())
    runtime = configparser.ConfigParser(interpolation=None)
    runtime.read_dict({section: dict(parser[section]) for section in parser.sections()})
    runtime.remove_option("database", "password")
    runtime["secrets"] = {"oracle_password_file": str(password_file), "master_key_file": str(master)}
    buffer = io.StringIO()
    runtime.write(buffer)
    effective = state_dir / "effective.ini"
    release.write_private_secret(effective, buffer.getvalue(), replace=effective.exists())
    values = release.deployment_values(effective)
    values["healthchecks"] = True
    release.validate_observability(values)
    return effective, values


def installation_identity(values: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_name": values["project_name"],
        "data_dir": str(values["data_dir"]),
        "database": {
            key: values["database_" + key]
            for key in ("host", "port", "service_name", "username")
        },
        "enabled_apps": sorted(values["enabled_apps"]),
        "embedding_dimension": values["embedding_dimension"],
        "image_source_commit": manifest["image_source_commit"],
        "docling_models_sha256": manifest["docling_models"]["manifest_sha256"],
        "images": [
            {"name": record["name"], "digest": record.get("digest", record.get("registry_digest"))}
            for record in manifest["images"]
        ],
        "master_key_sha256": release.sha256_file(values["master_key_file"]),
    }


def check_installation(state: dict[str, Any] | None, identity: dict[str, Any], parser: configparser.ConfigParser) -> str:
    if state is not None:
        if state.get("identity") != identity:
            raise release.ReleaseError("安装目标、App 范围、镜像版本或主密钥已变化；禁止当作重新部署，须执行受控升级或恢复")
        if state.get("phase") not in {"initialized", "ready"}:
            raise release.ReleaseError("上次初始化未完成；已停止自动重试 DDL，请检查安装记录并由 DBA 恢复")
        return "restart"
    if parser.getboolean("installation", "initialize_empty_schema", fallback=False):
        return "initialize"
    if parser.getboolean("installation", "adopt_existing_schema", fallback=False):
        return "adopt"
    raise release.ReleaseError("没有安装记录；首次空库安装需设置 initialize_empty_schema = true，已有同版本部署需显式接管并恢复原主密钥")


def run(command: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess:
    """不把数据库工具的原始输出写入终端，避免密码或连接串泄露。"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise release.ReleaseError("部署步骤执行超时；请检查容器状态与受控日志") from error
    if result.returncode:
        codes = sorted(set(re.findall(
            r"\b(?:ORA-\d{5}|DPY-\d{4}|DPI-\d{4}|INSTALLATION_ERROR=[A-Za-z]+)\b",
            result.stdout + "\n" + result.stderr,
        )))
        detail = "；错误标识：" + ", ".join(codes) if codes else ""
        raise release.ReleaseError(f"部署步骤失败（退出码 {result.returncode}{detail}）；请检查容器状态与受控日志")
    return result


def compose_command(output: Path, project_name: str) -> list[str]:
    return ["docker", "compose", "--project-name", project_name, "--project-directory", str(output), "-f", str(output / "compose.yaml")]


def guard_project(values: dict[str, Any], *, action: str) -> None:
    """防止新数据目录使用默认项目名误替换已有部署。"""
    if action != "initialize":
        return
    existing = run([
        "docker", "ps", "--all", "--filter",
        f"label=com.docker.compose.project={values['project_name']}", "--format", "{{.ID}}",
    ]).stdout.strip()
    if existing:
        raise release.ReleaseError("同名 Compose Project 已存在；请使用原数据目录或显式接管，禁止作为新安装替换")


def selected_images(values: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    catalog, _ = release.validate_catalog()
    selected = release.selected_service_configs(catalog, values)
    names = {item["name"] for item in catalog["images"] if item["id"] in selected}
    names.update({catalog["installer"]["name"], catalog["ui"]["name"]})
    return [record for record in manifest["images"] if record["name"] in names]


def acquire_images(root: Path, values: dict[str, Any], manifest: dict[str, Any]) -> None:
    offline = manifest["schema"] == release.OFFLINE_BUNDLE_SCHEMA
    records = selected_images(values, manifest)
    if offline:
        archive = release.safe_bundle_member(root, manifest["image_archive"])
        run(["docker", "image", "load", "--input", str(archive)])
    for record in records:
        if offline:
            result = run(["docker", "image", "inspect", record["image_id"]])
            if json.loads(result.stdout)[0]["Id"] != record["image_id"]:
                raise release.ReleaseError("离线镜像 Image ID 校验失败")
        else:
            reference = release.image_reference(values, record["name"])
            run(["docker", "image", "pull", "--platform", "linux/amd64", reference])
            details = json.loads(run(["docker", "image", "inspect", reference]).stdout)[0]
            if reference not in details.get("RepoDigests", []):
                raise release.ReleaseError("本地镜像与发布 Digest 不一致")
            if details.get("Architecture") != "amd64" or details.get("Os") != "linux":
                raise release.ReleaseError("镜像运行平台与发布清单不一致")


def installer(compose: list[str], command: list[str], *, timeout: int = 3600) -> None:
    run([*compose, "--profile", "tools", "run", "--rm", "--no-deps", "--pull", "never", "installer", *command], timeout=timeout)


def bootstrap_database(compose: list[str], values: dict[str, Any], *, adopt: bool) -> None:
    command = ["python", "/opt/kbot/scripts/db/initialize_installation.py", "--apps", *values["enabled_apps"]]
    if adopt:
        command.append("--check-only")
    installer(compose, command)


def verify_running(compose: list[str], expected: set[str]) -> dict[str, tuple[str, int]]:
    ids = run([*compose, "ps", "--all", "--quiet"]).stdout.split()
    if not ids:
        raise release.ReleaseError("没有发现运行中的业务容器")
    containers = json.loads(run(["docker", "inspect", *ids]).stdout)
    found: dict[str, tuple[str, int]] = {}
    for container in containers:
        service = container["Config"]["Labels"].get("com.docker.compose.service")
        if service not in expected:
            continue
        state = container["State"]
        if state.get("Status") != "running" or state.get("Health", {}).get("Status", "healthy") != "healthy":
            raise release.ReleaseError(f"容器尚未就绪：{service}")
        if service in found:
            raise release.ReleaseError(f"发现重复业务容器：{service}")
        found[service] = (container["Id"], container["RestartCount"])
    if set(found) != expected:
        raise release.ReleaseError("所选 App 的业务进程不完整")
    return found


def accept_deployment(compose: list[str], values: dict[str, Any]) -> None:
    catalog, topology = release.validate_catalog()
    configs = release.selected_service_configs(catalog, values)
    expected = {release.compose_name(item["process_key"]) for item in topology["processes"] if item["service_config"] in configs} | {"ui"}
    before = verify_running(compose, expected)
    print("服务已就绪，正在观察后台进程稳定性……", flush=True)
    time.sleep(10)
    if verify_running(compose, expected) != before:
        raise release.ReleaseError("稳定性观察期间有容器重启或被替换，部署验收未通过")
    for app_id in values["enabled_apps"]:
        path = catalog["app_packs"][app_id]["entry_path"]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{values['ui_port']}{path}", timeout=5) as response:
                if response.status != 200 or b"<html" not in response.read().lower():
                    raise release.ReleaseError("所选 App 的登录页不可用")
        except OSError as error:
            raise release.ReleaseError("用户入口访问失败，请检查入口端口和 Nginx") from error


def preflight() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise release.ReleaseError("当前只支持 Linux x86_64 主机")
    if not shutil.which("docker"):
        raise release.ReleaseError("请先安装 Docker Engine 和 Docker Compose v2")
    endpoint = os.environ.get("DOCKER_HOST")
    if not endpoint:
        endpoint = json.loads(run(["docker", "context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"], timeout=30).stdout)
    if not str(endpoint).startswith("unix://"):
        raise release.ReleaseError("部署使用本机目录挂载，不支持远程 Docker Context")
    run(["docker", "compose", "version"])
    help_text = run(["docker", "compose", "up", "--help"]).stdout
    if "--wait-timeout" not in help_text:
        raise release.ReleaseError("Docker Compose 版本过旧，请升级到支持 up --wait-timeout 的 v2 或更新版本")
    server = json.loads(run(["docker", "info", "--format", "{{json .}}"], timeout=30).stdout)
    if server.get("OSType") != "linux" or server.get("Architecture") not in {"x86_64", "amd64"}:
        raise release.ReleaseError("Docker 服务端必须运行在 Linux x86_64")


def deploy(root: Path) -> None:
    manifest = verify_package(root)
    config = root / "deployment.ini"
    if not config.exists():
        release.write_private_secret(config, (root / "deployment.ini.example").read_text(encoding="utf-8"))
        print(f"已创建唯一配置：{config}；请修改后再次执行 ./kbot-deploy")
        return
    parser, state_dir = load_input(config, manifest)
    preflight()
    with installation_lock(state_dir):
        state = read_state(state_dir)
        effective, values = prepare_config(parser, state_dir, state)
        identity = installation_identity(values, manifest)
        action = check_installation(state, identity, parser)
        guard_project(values, action=action)
        values["image_digests"] = {item["name"]: item.get("digest", item.get("registry_digest")) for item in manifest["images"]}
        if manifest["schema"] == release.OFFLINE_BUNDLE_SCHEMA:
            # docker save/load 不保证保留 RepoDigest，离线 Compose 直接锁定已校验的 Image ID。
            values["image_ids"] = {item["name"]: item["image_id"] for item in manifest["images"]}
        print("[1/6] 校验配置与发布工件完成", flush=True)
        output = state_dir / "deployments" / release.secrets.token_hex(8)
        release.render_deployment(effective, output=output, values=values)
        compose = compose_command(output, values["project_name"])
        run([*compose, "config", "--quiet"])
        observability = None
        if values["observability_enabled"]:
            if manifest["schema"] == release.OFFLINE_BUNDLE_SCHEMA:
                raise release.ReleaseError("核心离线包不含第三方观测镜像，请关闭 observability")
            module = release.load_aiops_stack_module()
            settings = module._load_settings(output / "observability/aiops-stack.ini")
            module._preflight(settings)
            observability = (module, settings)
            definition = json.loads(run([*module._compose_command(settings), "config", "--format", "json"]).stdout)
            services = module._selected_services(settings)
            if any(definition["services"][name].get("build") for name in services):
                raise release.ReleaseError("所选观测组件需要源码构建；请先发布预构建镜像，不在用户安装机执行 build")
        print("[2/6] 准备所选 App 的锁定镜像……", flush=True)
        acquire_images(root, values, manifest)
        if observability:
            module, settings = observability
            run([*module._compose_command(settings), "pull", *module._selected_services(settings)])
        print("[3/6] 安装并校验解析模型……", flush=True)
        if "knowledge_core" in release.selected_service_configs(release.validate_catalog()[0], values):
            release.install_offline_docling_models(root / "bundle-manifest.json", effective)
        if action != "restart":
            # 先确认 Installer 包含新入口；旧镜像在执行任何 DDL 前即拒绝。
            installer(compose, ["python", "/opt/kbot/scripts/db/initialize_installation.py", "--help"])
            state = {"schema": STATE_SCHEMA, "identity": identity, "phase": "initializing", "deployment_dir": str(output)}
            if action == "initialize":
                private_json(state_dir / "installation.json", state)
            print("[4/6] 校验已有安装" if action == "adopt" else "[4/6] 初始化已授权的空白 Schema", flush=True)
            bootstrap_database(compose, values, adopt=action == "adopt")
            state["phase"] = "initialized"
            private_json(state_dir / "installation.json", state)
        else:
            print("[4/6] 保留已有数据库、账号和主密钥，不重放初始化", flush=True)
        assert state is not None
        state["deployment_dir"] = str(output)
        state["phase"] = "initialized"
        private_json(state_dir / "installation.json", state)
        timeout = parser.getint("installation", "timeout_seconds", fallback=300)
        print("[5/6] 启动服务并等待就绪……", flush=True)
        run([*compose, "up", "--detach", "--remove-orphans", "--no-build", "--pull", "never", "--force-recreate", "--wait", "--wait-timeout", str(timeout)], timeout=timeout + 30)
        if observability:
            module, settings = observability
            run([*module._compose_command(settings), "up", "--detach", "--no-build", "--pull", "never", "--wait", "--wait-timeout", str(timeout), *module._selected_services(settings)], timeout=timeout + 30)
        print("[6/6] 验收入口和进程稳定性……", flush=True)
        accept_deployment(compose, values)
        state["phase"] = "ready"
        private_json(state_dir / "installation.json", state)
        print("安装成功，可以登录：")
        catalog, _ = release.validate_catalog()
        for app_id in values["enabled_apps"]:
            print(values["public_base_url"] + catalog["app_packs"][app_id]["entry_path"])
        accounts_path = state_dir / "initial-accounts.json"
        if accounts_path.is_file():
            print(f"首次账号交接文件：{accounts_path}")
        else:
            print("账号凭据沿用已有管理员；不会重新生成首次交接文件。")
        print(f"请受控备份主密钥：{state_dir / 'master_key'}；升级或恢复必须沿用。")
        print("模型 Provider 与业务资源需在 App 内配置；基础安装通过不代表模型或诊断业务已验收。")


def main() -> int:
    if sys.version_info < (3, 11):
        print("部署入口需要 Python 3.11+，不需要安装 Python 第三方依赖。", file=sys.stderr)
        return 2
    if len(sys.argv) != 2:
        print("请通过零参数 ./kbot-deploy 入口执行。", file=sys.stderr)
        return 2
    try:
        deploy(Path(sys.argv[1]).resolve())
    except (release.ReleaseError, OSError, ValueError, TypeError, KeyError, configparser.Error) as error:
        # 第三方异常可能携带连接串或配置值，只展示我们明确构造的安全错误。
        message = str(error) if isinstance(error, release.ReleaseError) else "配置或部署工件读取失败，请检查权限与格式"
        print(f"部署停止：{message}。未自动清库、回滚 DDL 或重置已有账号。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
