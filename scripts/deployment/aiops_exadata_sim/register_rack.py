"""按原厂允许的机柜包含关系接入 SIM 交换机/PDU，不直接创建 internal_contains。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-rack"
RACK = "AIOPS_SIM_EXADATA_RACK01"
SYSTEM = "AIOPS_SIM_EXADATA_DBM01"
HARDWARE = "AIOPS_SIM_EXADATA_SWITCH01:oracle_si_netswitch;AIOPS_SIM_EXADATA_PDU01:oracle_si_pdu"


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments],
                          capture_output=True, text=True, timeout=180)


def accepted(result):
    return result.returncode == 0 and not any(word in (result.stdout + result.stderr).lower()
        for word in ("warning", "error", "failed", "cannot", "not added"))


def arguments_for(phase):
    if phase == "create":
        return ["add_target", "-name=" + RACK, "-type=oracle_si_rack",
                "-display_name=SIM - Exadata Rack 01", "-timezone_region=UTC",
                "-properties=RackType:Generic 42U Cabinet;TotalSlots:42;EngineeredSystemId:" + SYSTEM]
    if phase == "members":
        return ["create_assoc", "-assoc_type=app_composite_contains",
                "-source=" + RACK + ":oracle_si_rack", "-dest=" + HARDWARE]
    return ["modify_system", "-name=" + SYSTEM, "-type=oracle_dbmachine",
            "-add_members=" + RACK + ":oracle_si_rack;" + HARDWARE]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("create", "members", "system"))
    phase = parser.parse_args().phase
    if os.geteuid() != 0:
        raise SystemExit("请以 root 使用现场安全 CLI 包装器。")
    STATE.mkdir(mode=0o700, exist_ok=True)
    if (STATE / (phase + "-started.json")).exists():
        raise SystemExit("已有阶段提交记录，保留现场，不重复或使用 force。")
    rack = cli("get_targets", "-targets=%:oracle_si_rack")
    exists = RACK in rack.stdout.split()
    if not accepted(rack) or (phase == "create") == exists:
        raise SystemExit("机柜发现失败或身份与阶段不符，不覆盖目标。")
    if phase != "create":
        previous = STATE / (("create" if phase == "members" else "members") + "-result.json")
        if not previous.is_file() or not json.loads(previous.read_text()).get("command_accepted"):
            raise SystemExit("缺少前序原厂成功凭证。")
    if phase == "members":
        for kind in ("oracle_si_netswitch", "oracle_si_pdu"):
            allowed = cli("list_allowed_pairs", "-source_type=oracle_si_rack", "-dest_type=" + kind)
            if not accepted(allowed) or "app_composite_contains" not in allowed.stdout:
                raise SystemExit("原厂未允许该机柜包含关系。")
    (STATE / (phase + "-started.json")).write_text(json.dumps({"phase": phase, "simulation": True}) + "\n")
    result = cli(*arguments_for(phase))
    (STATE / (phase + "-output.log")).write_text(result.stdout + result.stderr)
    (STATE / (phase + "-result.json")).write_text(json.dumps({"command_accepted": accepted(result),
        "native_exit_code": result.returncode, "inventory_verified": False, "simulation": True}) + "\n")
    print(result.stdout + result.stderr)
    if not accepted(result):
        raise SystemExit("原厂未确认成功，保留现场并独立核验，不重复提交。")


if __name__ == "__main__":
    main()
