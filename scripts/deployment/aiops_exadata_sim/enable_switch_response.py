"""仅启用固定 SIM 交换机默认禁用的管理访问点状态集合，原厂预检后应用。"""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-switch-response"


def arguments(preview):
    return ["modify_collection_schedule", "-targetType=oracle_si_netswitch",
        "-targetNames=AIOPS_SIM_EXADATA_SWITCH01", "-collectionName=AccessPointResponse",
        "-collectionStatus=ENABLED", "-freqType=MINUTE", "-freqValue=5", "-preview=" + preview]


def cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *args],
        capture_output=True, text=True, timeout=180)


def main():
    if os.geteuid() != 0 or STATE.exists():
        raise SystemExit("需要 root 且无既有提交记录，保留现场，不重复执行。")
    target = cli("get_targets", "-targets=AIOPS_SIM_EXADATA_SWITCH01:oracle_si_netswitch")
    if target.returncode or "AIOPS_SIM_EXADATA_SWITCH01" not in target.stdout.split():
        raise SystemExit("固定 SIM 目标不存在。")
    STATE.mkdir(mode=0o700)
    for phase, args in (("preview", arguments("Y")), ("apply", arguments("N")),
        ("collect", ["collect_metric", "-target_name=AIOPS_SIM_EXADATA_SWITCH01",
            "-target_type=oracle_si_netswitch", "-metric_name=AccessPointResponse"])):
        result = cli(*args)
        output = result.stdout + result.stderr
        (STATE / (phase + ".log")).write_text(output)
        accepted = result.returncode == 0 and not any(word in output.lower()
            for word in ("warning", "error", "failed", "not supported"))
        (STATE / (phase + ".json")).write_text(json.dumps({"command_accepted": accepted,
            "native_exit_code": result.returncode, "samples_verified": False}) + "\n")
        print(json.dumps({"phase": phase, "command_accepted": accepted, "samples_verified": False}))
        if not accepted:
            raise SystemExit("原厂未确认成功，保留现场；不使用 force 或修改原厂 XML。")


if __name__ == "__main__":
    main()
