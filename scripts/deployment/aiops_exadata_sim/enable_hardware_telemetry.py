"""仅调整固定 SIM 硬件集合，经原厂预检后启用；不修改 XML 或 OEM 仓库表。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path("/opt/aiops-oem-lab")
SCHEDULES = (
    ("oracle_si_netswitch", "AIOPS_SIM_EXADATA_SWITCH01", "oracle_si_hw_sensor_status", 5),
    ("oracle_si_netswitch", "AIOPS_SIM_EXADATA_SWITCH01", "oracle_si_switch_power_status", 5),
    ("oracle_si_netswitch", "AIOPS_SIM_EXADATA_SWITCH01", "oracle_si_switch_status", 5),
    ("oracle_si_pdu", "AIOPS_SIM_EXADATA_PDU01", "PduSummaryStatusCol", 15),
    ("oracle_si_pdu", "AIOPS_SIM_EXADATA_PDU01", "PduModuleStatusCol", 15),
    ("oracle_si_pdu", "AIOPS_SIM_EXADATA_PDU01", "PduModulePhaseStatusCol", 15),
)


def arguments(item, preview):
    kind, target, collection, interval = item
    return ["modify_collection_schedule", "-targetType=" + kind, "-targetNames=" + target,
        "-collectionName=" + collection, "-collectionStatus=ENABLED", "-freqType=MINUTE",
        "-freqValue=" + str(interval), "-preview=" + ("Y" if preview else "N")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--pdu-only", action="store_true", help="只预检/启用固定 PDU 三个集合")
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("需要 root 调用现场私有 CLI")
    record = Path(tempfile.mkdtemp(prefix="exadata-hardware-collections-", dir=ROOT / "state"))
    os.chmod(record, 0o700)
    # 所有集合必须先通过 preview；发生错误不继续应用后续集合，也不使用 force。
    for preview in ((True, False) if args.apply else (True,)):
        for index, item in enumerate(SCHEDULES):
            if args.pdu_only and item[0] != "oracle_si_pdu":
                continue
            result = subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments(item, preview)],
                capture_output=True, text=True, timeout=180)
            output = result.stdout + result.stderr
            phase = "preview" if preview else "apply"
            (record / f"{index}-{phase}.log").write_text(output)
            accepted = result.returncode == 0 and not any(word in output.lower()
                for word in ("error", "failed", "not supported", "warning"))
            print(json.dumps({"phase": phase, "target": item[1], "collection": item[2],
                "command_accepted": accepted, "samples_verified": False, "record": str(record)}), flush=True)
            if not accepted:
                raise SystemExit("原厂未确认命令成功，保留记录，不继续")


if __name__ == "__main__":
    main()
