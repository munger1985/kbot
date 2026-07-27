"""验收 4.0 App、配置和本地启停脚本的进程拓扑。"""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY = ROOT / "resources" / "topology.toml"


def load_processes() -> tuple[dict, ...]:
    payload = tomllib.loads(TOPOLOGY.read_text(encoding="utf-8"))
    return tuple(payload.get("processes") or ())


def check_process_topology() -> list[str]:
    processes = load_processes()
    errors: list[str] = []
    keys = [str(item.get("process_key")) for item in processes]
    modules = [str(item.get("module")) for item in processes]
    if len(keys) != len(set(keys)):
        errors.append("process_key 必须唯一")
    if len(modules) != len(set(modules)):
        errors.append("App module 必须唯一")

    discovered = {
        f"{path.parents[1].name}.entrypoints.{path.stem}"
        for path in (ROOT / "services").glob(
            "*/src/*/entrypoints/*.py"
        )
        if path.name != "__init__.py"
    }
    declared = set(modules)
    if discovered - declared:
        errors.append(f"拓扑缺少 App：{sorted(discovered - declared)}")
    if declared - discovered:
        errors.append(f"拓扑声明不存在 App：{sorted(declared - discovered)}")

    ports: dict[int, str] = {}
    service_names: dict[str, str] = {}
    start_script = (ROOT / "start_kbot.sh").read_text(encoding="utf-8")
    stop_script = (ROOT / "stop_kbot.sh").read_text(encoding="utf-8")
    if 'python -m "$module"' not in start_script:
        errors.append("start_kbot.sh 必须通过 python -m 启动 App 模块")
    if 'local module_name="$2"' not in stop_script:
        errors.append("stop_kbot.sh 必须支持按模块入口识别进程")
    for item in processes:
        key = str(item.get("process_key"))
        module = str(item.get("module"))
        kind = str(item.get("kind"))
        service_name = str(item.get("service_name") or "")
        if module not in start_script:
            errors.append(f"{key} 未被 start_kbot.sh 管理")
        if module not in stop_script:
            errors.append(f"{key} 未被 stop_kbot.sh 管理")

        if not service_name:
            errors.append(f"{key} 必须声明 service_name")
        else:
            previous_name = service_names.setdefault(service_name, key)
            if previous_name != key:
                errors.append(
                    f"服务身份 {service_name} 被 {previous_name} 与 {key} 重复使用"
                )
        if kind not in {"http", "worker"}:
            errors.append(f"{key} kind 必须为 http 或 worker")
        if kind == "http":
            port = item.get("port")
            if not isinstance(port, int):
                errors.append(f"{key} HTTP 进程必须声明整数端口")
                continue
            previous = ports.setdefault(port, key)
            if previous != key:
                errors.append(f"端口 {port} 被 {previous} 与 {key} 重复使用")
        elif "port" in item:
            errors.append(f"{key} Worker 不应声明监听端口")
    return errors


def main() -> int:
    errors = check_process_topology()
    if errors:
        print("KBot 进程拓扑校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    processes = load_processes()
    http_count = sum(item["kind"] == "http" for item in processes)
    print(
        "KBot 进程拓扑校验通过："
        f"{len(processes)} 个进程，{http_count} 个 HTTP，"
        f"{len(processes) - http_count} 个 Worker"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
