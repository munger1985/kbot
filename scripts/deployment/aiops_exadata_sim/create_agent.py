"""创建空卷上的标准 OEM Agent，专用于模拟 Cell；不修改既有 Agent。"""

import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-monitor"
NAME = "aiops-exadata-sim-monitor"
VOLUME = "aiops_exadata_sim_monitor_data"


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 运行调试部署。")
    for arguments in (("container", "inspect", NAME), ("volume", "inspect", VOLUME)):
        result = subprocess.run(["docker", *arguments], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            raise SystemExit("本次容器名或卷已存在，不接管、不覆盖。")
    entrypoint = Path(__file__).with_name("agent_entrypoint.py")
    if STATE.exists() or not entrypoint.is_file():
        raise SystemExit("状态目录已存在或缺少标准 Agent 运行入口，停止。")
    STATE.mkdir(mode=0o700)
    subprocess.run(["docker", "volume", "create", "--label", "aiops.simulation=exadata", VOLUME], check=True)
    subprocess.run([
        "docker", "run", "-d", "--name", NAME, "--hostname", "aiops-exadata-monitor.lab",
        "--label", "aiops.simulation=exadata", "--network", "aiops-oem-lab",
        "--network-alias", "aiops-exadata-monitor.lab", "--restart", "unless-stopped", "--init",
        "--memory", "4g", "--memory-swap", "8g", "--cpus", "2", "--shm-size", "512m",
        "--ulimit", "nproc=16384:16384", "--ulimit", "nofile=65536:65536",
        "--log-opt", "max-size=20m", "--log-opt", "max-file=3",
        "-v", VOLUME + ":/opt/oem-remote", "-v", str(STATE) + ":/opt/remote-state",
        "-v", str(ROOT / "staging/agent19") + ":/media/agent:ro",
        "-v", str(entrypoint) + ":/usr/local/bin/remote_agent_entrypoint.py:ro",
        "aiops-lab/oem-remote-agent:ol8",
    ], check=True)
    print(json.dumps({"container": NAME, "volume": VOLUME, "mode": "STANDARD", "published_ports": []}))


if __name__ == "__main__":
    main()
