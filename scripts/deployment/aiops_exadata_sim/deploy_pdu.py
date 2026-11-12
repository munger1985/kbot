"""仅在既有 OEM 实验网创建独立只读 PDU 管理模拟器，不修改既有设备。"""

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-pdu"
NAME = "aiops-exadata-pdu-sim"
ADDRESS = "172.22.0.10"


def checked(*arguments):
    return subprocess.check_output(arguments, text=True)


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 部署隔离模拟器。")
    if STATE.exists() or NAME in checked("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        raise SystemExit("状态或同名容器已存在，保留现场，不覆盖。")
    network = json.loads(checked("docker", "network", "inspect", "aiops-oem-lab"))[0]
    if any(item.get("IPv4Address", "").split("/")[0] == ADDRESS
           for item in network.get("Containers", {}).values()):
        raise SystemExit("模拟地址已被占用。")
    sources = [Path(__file__).resolve().with_name(name) for name in ("pdu_simulator.py", "switch_simulator.py")]
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise SystemExit("模拟器源码缺失或身份不符。")
        compile(source.read_bytes(), str(source), "exec")
    STATE.mkdir(mode=0o700)
    community = STATE / "snmp-community"
    with open(community, "x", opener=lambda path, flags: os.open(path, flags, 0o400)) as stream:
        stream.write(secrets.token_hex(24) + "\n")
    os.chown(community, 54321, 54321)
    mounts = []
    for source in sources:
        mounts.extend(["--mount", "type=bind,src=" + str(source) + ",dst=/sim/" + source.name + ",readonly"])
    container = checked("docker", "run", "-d", "--name", NAME, "--hostname", "aiops-exadata-pdu.lab",
        "--label", "aiops.simulation=exadata", "--network", "aiops-oem-lab", "--ip", ADDRESS,
        "--network-alias", "aiops-exadata-pdu.lab", "--restart", "unless-stopped",
        "--read-only", "--user", "54321:54321", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--sysctl", "net.ipv4.ip_unprivileged_port_start=0",
        "--memory", "64m", "--cpus", "0.25", "--pids-limit", "16",
        "--log-opt", "max-size=5m", "--log-opt", "max-file=2",
        "--mount", "type=bind,src=" + str(community) + ",dst=/run/secrets/snmp-community,readonly",
        *mounts, "--entrypoint", "/usr/bin/python3.12", "aiops-lab/oem-remote-agent:ol8",
        "-B", "/sim/pdu_simulator.py", "--community-file", "/run/secrets/snmp-community")
    receipt = {"container": NAME, "container_id": container.strip(), "address": ADDRESS,
               "simulation": True, "published_ports": [], "scope": "MANAGEMENT_REACHABILITY_ONLY",
               "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}}
    (STATE / "created.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
