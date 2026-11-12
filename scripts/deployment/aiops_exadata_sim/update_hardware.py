"""定向更新现有 SIM 交换机/PDU；预检摘要、只读挂载和隔离边界，备份后保留 inode。"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path("/opt/aiops-oem-lab")
FILES = {
    "switch": ("aiops-exadata-switch-sim", ROOT / "exadata-sim/switch_simulator.py", "/sim/switch_simulator.py", "switch_simulator.py"),
    "pdu": ("aiops-exadata-pdu-sim", ROOT / "exadata-sim-pdu/pdu_simulator.py", "/sim/pdu_simulator.py", "pdu_simulator.py"),
    "pdu_codec": ("aiops-exadata-pdu-sim", ROOT / "exadata-sim-pdu/switch_simulator.py", "/sim/switch_simulator.py", "switch_simulator.py"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(info, container, target, destination):
    if info.get("Name") != "/" + container or not info.get("State", {}).get("Running"):
        raise RuntimeError("只允许更新正在运行的固定 SIM 容器")
    host, config = info["HostConfig"], info["Config"]
    if not host.get("ReadonlyRootfs") or not config.get("User") or config["User"] in ("0", "root"):
        raise RuntimeError("需要非 root 与只读根文件系统")
    if host.get("Privileged") or host.get("CapAdd") or host.get("PortBindings"):
        raise RuntimeError("拒绝特权、额外能力或发布宿主端口")
    mounts = [m for m in info["Mounts"] if m["Destination"] == destination]
    if len(mounts) != 1 or mounts[0]["Type"] != "bind" or mounts[0]["RW"] or Path(mounts[0]["Source"]) != target:
        raise RuntimeError("源码必须来自固定路径的唯一只读文件绑定")


def update(source, expected, apply=False):
    if os.geteuid() != 0 or set(expected) != set(FILES):
        raise RuntimeError("需要 root 及全部三个已核对摘要")
    plan = []
    containers = {}
    for key, (container, target, destination, name) in FILES.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected[key]) or target.is_symlink() or not target.is_file():
            raise RuntimeError("摘要或目标文件无效")
        if container not in containers:
            info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))
            if len(info) != 1:
                raise RuntimeError("容器身份不唯一")
            containers[container] = info[0]
        validate(containers[container], container, target, destination)
        previous, payload = target.read_bytes(), (source / name).read_bytes()
        if digest(previous) != expected[key]:
            raise RuntimeError("现场源码漂移，拒绝覆盖")
        compile(payload, name, "exec")
        plan.append((key, container, target, destination, previous, payload, target.stat().st_ino))
    receipt = {"applied": False, "updated_at": datetime.now(UTC).isoformat(), "files": {
        key: {"old": digest(previous), "new": digest(payload)}
        for key, _, _, _, previous, payload, _ in plan}}
    if not apply:
        return receipt
    backup = Path(tempfile.mkdtemp(prefix="exadata-hardware-update-", dir=ROOT / "state"))
    os.chmod(backup, 0o700)
    for key, _, target, _, _, _, _ in plan:
        shutil.copy2(target, backup / (key + ".py"))
    receipt["backup"] = str(backup)
    (backup / "receipt.json").write_text(json.dumps(receipt) + "\n")
    try:
        # 只原位更新文件；原子 rename 会让运行容器仍绑定旧 inode。
        for _, container, target, destination, previous, payload, inode in plan:
            if target.read_bytes() != previous or target.stat().st_ino != inode:
                raise RuntimeError("预检后文件发生变化")
            target.write_bytes(payload)
            live = subprocess.check_output(["docker", "exec", container, "sha256sum", destination], text=True).split()[0]
            if target.stat().st_ino != inode or live != digest(payload):
                raise RuntimeError("绑定文件更新验证失败")
        for container in containers:
            subprocess.run(["docker", "restart", "--time", "10", container], check=True, stdout=subprocess.DEVNULL)
            info = json.loads(subprocess.check_output(["docker", "inspect", container], text=True))[0]
            if not info["State"]["Running"]:
                raise RuntimeError("模拟器未恢复运行")
        receipt["applied"] = True
    except Exception:
        # 原位恢复备份，只重启本次两个模拟器；保留记录，错误继续向调用者抛出。
        for _, _, target, _, previous, _, _ in plan:
            target.write_bytes(previous)
        for container in containers:
            subprocess.run(["docker", "restart", "--time", "10", container], check=False, stdout=subprocess.DEVNULL)
        raise
    finally:
        (backup / "receipt.json").write_text(json.dumps(receipt) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent)
    parser.add_argument("--expected-switch", required=True)
    parser.add_argument("--expected-pdu", required=True)
    parser.add_argument("--expected-pdu-codec", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(update(args.source, {"switch": args.expected_switch, "pdu": args.expected_pdu,
        "pdu_codec": args.expected_pdu_codec}, args.apply)))


if __name__ == "__main__":
    main()
