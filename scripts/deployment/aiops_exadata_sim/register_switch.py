"""使用原厂 Access Point 注册隔离 SIM 交换机；不写 OEM 仓库或应用绑定表。"""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-switch-snmp"
TARGET = "AIOPS_SIM_EXADATA_SWITCH01"
ACCESS_POINT = TARGET + "_SNMP"
COMMAND_FILE = "/opt/oem/admin/exadata-switch-sim/add-switch.args"


def cli(*arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/oem_cli.py"), *arguments],
                          capture_output=True, text=True, timeout=180)


def arguments_for(community):
    if not re.fullmatch(r"[0-9a-f]{48}", community):
        raise ValueError("模拟器 community 格式不符，不生成原厂命令文件。")
    return ["add_target", "-name=" + TARGET, "-type=oracle_si_netswitch",
            "-host=aiops-exadata-monitor.lab", "-display_name=SIM - Exadata Switch 01",
            "-access_point_name=" + ACCESS_POINT, "-access_point_type=oracle_si_switch_cisco_ios_snmp",
            "-properties=dispatch.url->snmp://172.22.0.9:1161;SNMPPort->1161;SNMPVersion->v2c;EnableFabricTarget->false",
            "-subseparator=properties=->",
            "-monitoring_cred=snmp_v1v2_v3;oracle_si_switch_cisco_ios_snmp;SNMPV1Creds;COMMUNITY:" + community,
            "-timezone_region=UTC", "-timeout=120"]


def command_file_for(community):
    # 原厂 input_file 不替换 monitoring_cred 内的标签。使用原厂 argfile，
    # 只让临时 0600 文件与 EMCLI 内存持有 community，不将其放入进程参数。
    arguments = arguments_for(community)
    options = [name + '=\"' + value + '\"' for name, value in (argument.split("=", 1) for argument in arguments[1:])]
    return arguments[0] + " " + " ".join(options) + "\n"


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 使用现场安全 CLI 包装器。")
    started = STATE / "target-started.json"
    if started.exists() or not (STATE / "created.json").is_file():
        raise SystemExit("缺少模拟容器凭证或已提交目标；保留现场，不重复注册。")
    for kind in ("oracle_si_netswitch", "oracle_si_switch_cisco_ios_snmp"):
        targets = cli("get_targets", "-targets=%:" + kind)
        if targets.returncode or TARGET in targets.stdout:
            raise SystemExit("目标发现失败或同名 SIM 目标已存在，不接管。")
    deployment = cli("get_plugin_deployment_status", "-plugin=oracle.sysman.si")
    block = next((part for part in deployment.stdout.split("Destination")
                  if part.lstrip().startswith(": Management Agent - aiops-exadata-monitor.lab:3872")), "")
    if deployment.returncode or "Status               : Success" not in block:
        raise SystemExit("独立 Agent 的 SI 插件尚未部署成功。")
    receipt = json.loads((STATE / "created.json").read_text())
    source = Path(__file__).resolve().with_name("switch_simulator.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
        raise SystemExit("模拟器源码摘要发生漂移。")
    inspection = json.loads(subprocess.check_output(["docker", "inspect", "aiops-exadata-switch-sim"], text=True))[0]
    if (not inspection["State"]["Running"] or inspection["HostConfig"]["PortBindings"]
            or inspection["Config"]["Labels"].get("aiops.simulation") != "exadata"
            or inspection["NetworkSettings"]["Networks"]["aiops-oem-lab"]["IPAddress"] != "172.22.0.9"):
        raise SystemExit("模拟容器身份、运行状态或网络不符。")
    community = (STATE / "snmp-community").read_text().strip()
    command_text = command_file_for(community)
    prepare = """import os,sys
from pathlib import Path
p=Path('/opt/oem/admin/exadata-switch-sim')
p.mkdir(mode=0o700,exist_ok=True)
os.chown(p,54321,54321)
f=p/'add-switch.args'
with open(f,'x',opener=lambda name,flags:os.open(name,flags,0o600)) as stream:
    stream.write(sys.stdin.read())
os.chown(f,54321,54321)
"""
    subprocess.run(["docker", "exec", "-i", "aiops-oem", "/usr/bin/python3.12", "-c", prepare],
                   input=command_text, text=True, check=True)
    started.write_text(json.dumps({"target": TARGET, "access_point": ACCESS_POINT, "simulation": True}) + "\n")
    try:
        result = cli("argfile", COMMAND_FILE)
        output = (result.stdout + result.stderr).replace(community, "<redacted>")
        accepted = result.returncode == 0 and not any(word in output.lower() for word in ("warning", "error", "failed"))
        (STATE / "target-output.log").write_text(output)
        (STATE / "target-result.json").write_text(json.dumps({"target": TARGET, "command_accepted": accepted,
            "native_exit_code": result.returncode, "inventory_verified": False, "simulation": True}) + "\n")
        print(output)
        if not accepted:
            raise SystemExit("原厂未确认成功；保留记录，先独立核验，不使用 force。")
    finally:
        subprocess.run(["docker", "exec", "aiops-oem", "/usr/bin/python3.12", "-c",
                        "from pathlib import Path; Path(" + repr(COMMAND_FILE) + ").unlink()"], check=True)


if __name__ == "__main__":
    main()
