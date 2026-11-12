"""使用原厂非 MAP 目标注册 SIM PDU；凭据只经临时私有 argfile 传递。"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-pdu"
TARGET = "AIOPS_SIM_EXADATA_PDU01"
COMMAND_FILE = "/opt/oem/admin/exadata-pdu-sim/add-pdu.args"


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments],
                          capture_output=True, text=True, timeout=180)


def arguments_for(community):
    if not re.fullmatch(r"[0-9a-f]{48}", community):
        raise ValueError("私有 community 格式无效。")
    # 不预设 HttpCommunicationProtocol；原厂必须实际 TCP 探测才产生可达性值。
    return ["add_target", "-name=" + TARGET, "-type=oracle_si_pdu",
            "-host=aiops-exadata-monitor.lab", "-display_name=SIM - Exadata PDU management only",
            "-properties=dispatch.url->http://172.22.0.10;SNMPPort->1161;SkipSnmpSubscription->true",
            "-subseparator=properties=->", "-timezone_region=UTC", "-timeout=120",
            "-monitoring_cred=snmp_v1v2_v3;oracle_si_pdu;SNMPV1Creds;COMMUNITY:" + community]


def command_file_for(community):
    args = arguments_for(community)
    return args[0] + " " + " ".join(name + '=\"' + value + '\"'
        for name, value in (arg.split("=", 1) for arg in args[1:])) + "\n"


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 使用现场安全 CLI 包装器。")
    if (STATE / "target-started.json").exists() or not (STATE / "created.json").is_file():
        raise SystemExit("缺少创建凭证或已有提交记录，保留现场。")
    targets = cli("get_targets", "-targets=%:oracle_si_pdu")
    if targets.returncode or TARGET in targets.stdout:
        raise SystemExit("目标发现失败或目标已存在，不接管。")
    deployment = cli("get_plugin_deployment_status", "-plugin=oracle.sysman.si")
    block = next((part for part in deployment.stdout.split("Destination")
                  if part.lstrip().startswith(": Management Agent - aiops-exadata-monitor.lab:3872")), "")
    if deployment.returncode or "Status               : Success" not in block:
        raise SystemExit("独立 Agent 的 SI 插件尚未部署成功。")
    receipt = json.loads((STATE / "created.json").read_text())
    if set(receipt["source_sha256"]) != {"pdu_simulator.py", "switch_simulator.py"}:
        raise SystemExit("模拟器源码凭证不完整。")
    for name, expected in receipt["source_sha256"].items():
        if name not in {"pdu_simulator.py", "switch_simulator.py"}:
            raise SystemExit("模拟器源码凭证无效。")
        source = Path(__file__).resolve().with_name(name)
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise SystemExit("模拟器源码发生漂移。")
    inspection = json.loads(subprocess.check_output(["docker", "inspect", "aiops-exadata-pdu-sim"], text=True))[0]
    if (not inspection["State"]["Running"] or inspection["HostConfig"]["PortBindings"]
            or inspection["Config"]["Labels"].get("aiops.simulation") != "exadata"
            or inspection["NetworkSettings"]["Networks"]["aiops-oem-lab"]["IPAddress"] != "172.22.0.10"):
        raise SystemExit("SIM 容器身份、状态或网络不符。")
    community = (STATE / "snmp-community").read_text().strip()
    prepare = """import os,sys
from pathlib import Path
p=Path('/opt/oem/admin/exadata-pdu-sim')
p.mkdir(mode=0o700,exist_ok=True)
os.chown(p,54321,54321)
f=p/'add-pdu.args'
with open(f,'x',opener=lambda name,flags:os.open(name,flags,0o600)) as stream:
    stream.write(sys.stdin.read())
os.chown(f,54321,54321)
"""
    subprocess.run(["docker", "exec", "-i", "aiops-oem", "/usr/bin/python3.12", "-c", prepare],
                   input=command_file_for(community), text=True, check=True)
    (STATE / "target-started.json").write_text(json.dumps({"target": TARGET, "simulation": True}) + "\n")
    try:
        result = cli("argfile", COMMAND_FILE)
        output = (result.stdout + result.stderr).replace(community, "<redacted>")
        accepted = result.returncode == 0 and not any(word in output.lower()
            for word in ("warning", "error", "failed", "cannot"))
        (STATE / "target-output.log").write_text(output)
        (STATE / "target-result.json").write_text(json.dumps({"target": TARGET,
            "command_accepted": accepted, "native_exit_code": result.returncode,
            "inventory_verified": False, "simulation": True}) + "\n")
        print(output)
        if not accepted:
            raise SystemExit("原厂未确认成功，保留现场，不使用 force。")
    finally:
        subprocess.run(["docker", "exec", "aiops-oem", "/usr/bin/python3.12", "-c",
                        "from pathlib import Path; Path(" + repr(COMMAND_FILE) + ").unlink()"], check=True)


if __name__ == "__main__":
    main()
