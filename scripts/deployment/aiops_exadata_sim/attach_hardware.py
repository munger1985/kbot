"""只向 SIM 整机追加专用实验 DB Server 主机；交换机/PDU 必须经 register_rack。"""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-hardware-members"
SYSTEM = "AIOPS_SIM_EXADATA_DBM01"
MEMBERS = (("aiops-exadata-monitor.lab", "host"),)


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments],
                          capture_output=True, text=True, timeout=180)


def accepted(result):
    return result.returncode == 0 and not any(word in (result.stdout + result.stderr).lower()
        for word in ("warning", "error", "failed", "cannot", "not added"))


def main():
    if os.geteuid() != 0 or STATE.exists():
        raise SystemExit("需要 root 且无既有提交记录；保留现场，不重复执行。")
    for name in ("aiops-exadata-sim-monitor",):
        data = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        if not data["State"]["Running"] or data["Config"]["Labels"].get("aiops.simulation") != "exadata":
            raise SystemExit("专用 SIM 容器身份或状态不符。")
    for name, kind in MEMBERS:
        result = cli("get_targets", "-targets=" + name + ":" + kind)
        if not accepted(result) or not any(kind in row.split() and name in row.split()
                and "Up" in row.split() for row in result.stdout.splitlines()):
            raise SystemExit("固定 SIM 目标不存在或原厂未确认 Up。")
    before = cli("get_system_members", "-name=" + SYSTEM, "-type=oracle_dbmachine", "-depth=2")
    expected = {(SYSTEM, "AIOPS_SIM_EXADATA_GRID01", "oracle_exadata_grid", "1"),
                ("AIOPS_SIM_EXADATA_GRID01", "AIOPS_SIM_EXADATA_CELL01", "oracle_exadata", "2")}
    rows = {tuple(row.split()[:4]) for row in before.stdout.splitlines()
            if row.startswith("AIOPS_SIM_EXADATA_")}
    if not accepted(before) or rows != expected:
        raise SystemExit("整机原始成员与审核基线不符，停止，不推断或接管关系。")
    STATE.mkdir(mode=0o700)
    (STATE / "before.log").write_text(before.stdout)
    (STATE / "started.json").write_text(json.dumps({"system": SYSTEM, "members": MEMBERS,
        "simulation": True, "host_scope": "DEDICATED_SIM_AGENT_HOST_NOT_PHYSICAL_DB_SERVER"}) + "\n")
    result = cli("modify_system", "-name=" + SYSTEM, "-type=oracle_dbmachine",
                 "-add_members=" + ";".join(name + ":" + kind for name, kind in MEMBERS))
    (STATE / "native.log").write_text(result.stdout + result.stderr)
    after = cli("get_system_members", "-name=" + SYSTEM, "-type=oracle_dbmachine", "-depth=2")
    (STATE / "after.log").write_text(after.stdout + after.stderr)
    rows = {tuple(row.split()[:4]) for row in after.stdout.splitlines()
            if row.startswith(("AIOPS_SIM_EXADATA_", SYSTEM))}
    verified = accepted(result) and accepted(after) and rows == expected | {
        (SYSTEM, name, kind, "1") for name, kind in MEMBERS}
    receipt = {"command_accepted": accepted(result), "native_members_verified": verified,
               "application_bindings_confirmed": False, "simulation": True}
    (STATE / "result.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))
    print(after.stdout)
    if not verified:
        raise SystemExit("原厂关系独立核验未通过，保留现场；不自动重试或删除成员。")


if __name__ == "__main__":
    main()
