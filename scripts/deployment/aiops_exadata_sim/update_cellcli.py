"""仅更新已存在的实验 CellCLI 绑定文件；校验漂移、备份并保留绑定 inode，不重启容器。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


CONTAINER = "aiops-exadata-cellcli-sim"
TARGET = Path("/opt/aiops-oem-lab/exadata-sim/cell_simulator.py")
STATE = Path("/opt/aiops-oem-lab/state")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_mount(container, target):
    if container.get("Name") != "/" + CONTAINER or not container.get("State", {}).get("Running"):
        raise RuntimeError("只允许更新正在运行的专用 CellCLI 模拟器")
    mounts = [item for item in container.get("Mounts", []) if item.get("Destination") == "/sim/cell_simulator.py"]
    if len(mounts) != 1 or mounts[0].get("Type") != "bind" or mounts[0].get("RW") is not False:
        raise RuntimeError("模拟器必须使用唯一的只读源码绑定")
    if Path(mounts[0].get("Source", "")) != target:
        raise RuntimeError("绑定源路径与限定模拟文件不符")


def update(source, expected_sha256):
    if os.geteuid() != 0:
        raise RuntimeError("请以 root 更新隔离模拟器")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise RuntimeError("必须提供已核对的现场文件 SHA256")
    if TARGET.is_symlink() or not TARGET.is_file() or source.resolve() == TARGET:
        raise RuntimeError("目标文件异常或新旧文件为同一文件")
    payload = source.read_bytes()
    compile(payload, "cell_simulator.py", "exec")
    info = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER], text=True))
    if len(info) != 1:
        raise RuntimeError("容器身份不唯一")
    validate_mount(info[0], TARGET)
    previous = TARGET.read_bytes()
    if digest(previous) != expected_sha256:
        raise RuntimeError("现场文件已漂移，停止；不覆盖未知修改")
    if previous == payload:
        return {"changed": False, "sha256": digest(payload)}
    backup = Path(tempfile.mkdtemp(prefix="exadata-sim-update-", dir=STATE))
    shutil.copy2(TARGET, backup / "cell_simulator.py")
    inode = TARGET.stat().st_ino
    receipt = {"container": CONTAINER, "target": str(TARGET), "backup": str(backup),
               "old_sha256": digest(previous), "new_sha256": digest(payload),
               "updated_at": datetime.now(timezone.utc).isoformat(), "restart": False}
    try:
        # 文件绑定固定 inode；替换路径会让容器继续读旧文件，因此原位更新并校验。
        TARGET.write_bytes(payload)
        if TARGET.stat().st_ino != inode:
            raise RuntimeError("绑定文件 inode 变化")
        live = subprocess.check_output(["docker", "exec", CONTAINER, "sha256sum", "/sim/cell_simulator.py"],
                                       text=True).split()[0]
        if live != digest(payload):
            raise RuntimeError("容器内源码摘要未更新")
    except Exception:
        TARGET.write_bytes(previous)
        raise
    (backup / "receipt.json").write_text(json.dumps(receipt) + "\n")
    return {"changed": True, **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("cell_simulator.py"))
    args = parser.parse_args()
    print(json.dumps(update(args.source, args.expected_sha256)))


if __name__ == "__main__":
    main()
