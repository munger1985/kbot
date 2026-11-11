"""创建使用独立公钥和强制只读命令的 CellCLI 模拟容器；不复用生产 SSH 身份。"""

import json
import os
from pathlib import Path
import subprocess


ROOT = Path("/opt/aiops-oem-lab")
STATE = ROOT / "state/exadata-sim-cellcli"
AGENT = "aiops-exadata-sim-monitor"
NAME = "aiops-exadata-cellcli-sim"
ADDRESS = "172.22.0.8"


def checked(*arguments):
    return subprocess.check_output(arguments, text=True)


def main():
    if os.geteuid() != 0:
        raise SystemExit("请以 root 部署隔离模拟器。")
    if STATE.exists() or NAME in checked("docker", "ps", "-a", "--format", "{{.Names}}").splitlines():
        raise SystemExit("模拟器状态或同名容器已存在，不覆盖。")
    network = json.loads(checked("docker", "network", "inspect", "aiops-oem-lab"))[0]
    if any(item.get("IPv4Address", "").split("/")[0] == ADDRESS
           for item in network.get("Containers", {}).values()):
        raise SystemExit("模拟地址已被占用。")
    directory = Path(__file__).resolve().parent
    for name in ("cellcli_entrypoint.py", "cell_simulator.py", "sshd_config"):
        if not (directory / name).is_file():
            raise SystemExit("缺少模拟器部署文件。")
    if subprocess.run(["docker", "exec", AGENT, "test", "-e", "/home/oracle/.ssh"]).returncode == 0:
        raise SystemExit("新 Agent 已有 SSH 配置，停止；不接管或覆盖任何密钥。")
    STATE.mkdir(mode=0o700)
    ssh = STATE / "ssh"
    ssh.mkdir(mode=0o755)
    checked("docker", "exec", AGENT, "install", "-d", "-o", "oracle", "-g", "oinstall", "-m", "700",
            "/opt/oem-remote/sim-ssh")
    checked("docker", "exec", AGENT, "ln", "-s", "/opt/oem-remote/sim-ssh", "/home/oracle/.ssh")
    checked("docker", "exec", "--user", "oracle", AGENT, "ssh-keygen", "-q", "-t", "ed25519", "-N", "",
            "-f", "/opt/oem-remote/sim-ssh/id_ed25519")
    public = checked("docker", "exec", AGENT, "cat", "/opt/oem-remote/sim-ssh/id_ed25519.pub").strip()
    if not public.startswith("ssh-ed25519 "):
        raise SystemExit("新生成的公钥格式不符。")
    authorized = ssh / "authorized_keys"
    authorized.write_text("restrict " + public + "\n")
    authorized.chmod(0o444)
    checked("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(ssh / "ssh_host_ed25519_key"))
    # 从本次生成的宿主公钥固定身份，不接受首次连接时从网络学习未知密钥。
    host_public = (ssh / "ssh_host_ed25519_key.pub").read_text().split()
    pinned = ADDRESS + ",aiops-exadata-cellcli.lab " + " ".join(host_public[:2]) + "\n"
    subprocess.run(["docker", "exec", "-i", "--user", "oracle", AGENT, "sh", "-c",
                    "umask 077; test ! -e /opt/oem-remote/sim-ssh/known_hosts && "
                    "dd of=/opt/oem-remote/sim-ssh/known_hosts status=none"],
                   input=pinned, text=True, check=True)
    result = checked("docker", "run", "-d", "--name", NAME, "--hostname", "aiops-exadata-cellcli.lab",
        "--label", "aiops.simulation=exadata", "--network", "aiops-oem-lab", "--ip", ADDRESS,
        "--network-alias", "aiops-exadata-cellcli.lab", "--restart", "unless-stopped", "--memory", "128m",
        "--cpus", "0.5", "--pids-limit", "64", "--log-opt", "max-size=5m", "--log-opt", "max-file=2",
        "--mount", "type=bind,src=" + str(ssh) + ",dst=/run/ssh,readonly",
        "--mount", "type=bind,src=" + str(directory / "cell_simulator.py") + ",dst=/sim/cell_simulator.py,readonly",
        "--mount", "type=bind,src=" + str(directory / "cellcli_entrypoint.py") + ",dst=/sim/cellcli_entrypoint.py,readonly",
        "--mount", "type=bind,src=" + str(directory / "sshd_config") + ",dst=/sim/sshd_config,readonly",
        "--entrypoint", "/usr/bin/python3.12", "aiops-lab/oem-remote-agent:ol8", "-B", "/sim/cellcli_entrypoint.py")
    receipt = {"container": NAME, "container_id": result.strip(), "address": ADDRESS,
               "simulation": True, "ssh_identity": "DEDICATED_ED25519", "published_ports": []}
    (STATE / "created.json").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
