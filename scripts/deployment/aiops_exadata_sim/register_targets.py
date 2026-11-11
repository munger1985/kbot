"""按 Cell → Grid → DB Machine 注册模拟目标；只用原厂 CLI，不改 OEM 内部表。"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-targets"
CELL = "AIOPS_SIM_EXADATA_CELL01"
GRID = "AIOPS_SIM_EXADATA_GRID01"
SYSTEM = "AIOPS_SIM_EXADATA_DBM01"
MASTER = "https://aiops-exadata-monitor.lab:3872/emd/main/"
TARGETS = {"cell": (CELL, "oracle_exadata"), "grid": (GRID, "oracle_exadata_grid"),
           "system": (SYSTEM, "oracle_dbmachine")}


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def native_success(result):
    """原厂可在退出码 0 时警告成员未加入；这种结果不能签发成功凭证。"""
    output = result.stdout.lower()
    return result.returncode == 0 and not any(
        word in output for word in ("warning", "error", "cannot", "failed", "not added"))


def arguments_for(phase):
    name, kind = TARGETS[phase]
    if phase == "cell":
        return ["add_target", "-name=" + name, "-type=" + kind,
                "-host=aiops-exadata-monitor.lab", "-display_name=SIM - Exadata Cell 01",
                "-timezone_region=UTC", "-properties=CellName:sim-cell01;MgmtIPAddr:172.22.0.8;SupportExacli:0",
                "-timeout=120"]
    child, child_kind = TARGETS["cell" if phase == "grid" else "grid"]
    props = "-master_agent=" + MASTER
    if phase == "system":
        props = "-properties=DBMachineID:" + name + " " + props
    return ["create_system", "-name=" + name, "-type=" + kind, "-meta_ver=241100.0",
            "-timezone_region=UTC", "-custom_props=" + props, "-add_members=" + child + ":" + child_kind]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=tuple(TARGETS))
    phase = parser.parse_args().phase
    if os.geteuid() != 0:
        raise SystemExit("请以 root 使用现场安全 CLI 包装器。")
    STATE.mkdir(mode=0o700, exist_ok=True)
    STATE.chmod(0o700)
    name, kind = TARGETS[phase]
    started = STATE / (phase + "-started.json")
    if started.exists():
        raise SystemExit("已有提交记录，停止；不重复注册、接管目标或使用 force。")
    targets = cli("get_targets", "-targets=%:" + kind)
    if targets.returncode or name in targets.stdout:
        raise SystemExit("目标发现失败或同名目标已存在，不接管未知对象。")
    if phase == "cell":
        plugin = cli("get_plugin_deployment_status", "-plugin=oracle.sysman.xa")
        deployment = next((block for block in plugin.stdout.split("Destination")
                           if block.lstrip().startswith(": Management Agent - aiops-exadata-monitor.lab:3872")), "")
        if plugin.returncode or "Status               : Success" not in deployment:
            raise SystemExit("独立 Agent 的 Exadata 插件尚未成功部署。")
    else:
        previous = STATE / (("cell" if phase == "grid" else "grid") + "-result.json")
        if not previous.is_file() or json.loads(previous.read_text()).get("command_accepted") is not True:
            raise SystemExit("缺少本次前序成功凭证；不注册依赖对象。")
    started.write_text(json.dumps({"target": name, "type": kind, "simulation": True,
                                   "started_at": datetime.now(timezone.utc).isoformat()}) + "\n")
    result = cli(*arguments_for(phase))
    accepted = native_success(result)
    (STATE / (phase + "-output.log")).write_text(result.stdout)
    (STATE / (phase + "-result.json")).write_text(json.dumps({
        "target": name, "type": kind, "native_exit_code": result.returncode,
        "command_accepted": accepted, "inventory_verified": False,
        "finished_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
    }) + "\n")
    print(result.stdout)
    if not accepted:
        raise SystemExit("原厂返回失败或警告；保留现场，独立核验，不据退出码宣称成员成功。")
    print("命令已接受；仍须通过只读成员接口核验，不能据此宣称页面验收通过。")


if __name__ == "__main__":
    main()
