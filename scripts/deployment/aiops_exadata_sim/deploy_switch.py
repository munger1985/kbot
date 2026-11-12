"""仅在既有 OEM 隔离实验网创建独立只读 SNMP 交换机，不修改 Oracle 软件。"""

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-switch-snmp"
NAME = "aiops-exadata-switch-sim"
ADDRESS = "172.22.0.9"


def checked(*arguments):
    return subprocess.check_output(arguments, text=True)


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 部署隔离模拟器。")
    if STATE.exists() or NAME in checked("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        raise SystemExit("模拟器状态或同名容器已存在；保留现场，不覆盖。")
    network = json.loads(checked("docker", "network", "inspect", "aiops-oem-lab"))[0]
    if any(item.get("IPv4Address", "").split("/")[0] == ADDRESS
           for item in network.get("Containers", {}).values()):
        raise SystemExit("模拟地址已被占用。")
    source = Path(__file__).resolve().with_name("switch_simulator.py")
    if source.is_symlink() or not source.is_file():
        raise SystemExit("缺少已审核的模拟器源码。")
    compile(source.read_bytes(), str(source), "exec")
    STATE.mkdir(mode=0o700)
    community = STATE / "snmp-community"
    with open(community, "x", opener=lambda path, flags: os.open(path, flags, 0o400)) as stream:
        stream.write(secrets.token_hex(24) + "\n")
    # 仅独立容器的非特权 oracle UID 可读；不使用公开 community。
    os.chown(community, 54321, 54321)
    container = checked("docker", "run", "-d", "--name", NAME, "--hostname", "aiops-exadata-switch.lab",
        "--label", "aiops.simulation=exadata", "--network", "aiops-oem-lab", "--ip", ADDRESS,
        "--network-alias", "aiops-exadata-switch.lab", "--restart", "unless-stopped",
        "--read-only", "--user", "54321:54321", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "64m", "--cpus", "0.25", "--pids-limit", "16",
        "--log-opt", "max-size=5m", "--log-opt", "max-file=2",
        "--mount", "type=bind,src=" + str(community) + ",dst=/run/secrets/snmp-community,readonly",
        "--mount", "type=bind,src=" + str(source) + ",dst=/sim/switch_simulator.py,readonly",
        "--entrypoint", "/usr/bin/python3.12", "aiops-lab/oem-remote-agent:ol8",
        "-B", "/sim/switch_simulator.py", "--community-file", "/run/secrets/snmp-community")
    receipt = {"container": NAME, "container_id": container.strip(), "address": ADDRESS,
               "simulation": True, "published_ports": [], "protocol": "SNMPv2c",
               "snmp_port": 1161, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (STATE / "created.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
