"""隔离 SSH CellCLI 模拟入口；强制命令只解析已审核 LIST，不执行远程 Shell。"""

import os
from pathlib import Path
import pwd
import shlex
import subprocess
import sys

from cell_simulator import fault, render


def main():
    if sys.argv[1:] == ["--command"]:
        try:
            arguments = shlex.split(os.environ.get("SSH_ORIGINAL_COMMAND", ""))
        except ValueError:
            arguments = []
        if len(arguments) != 4 or arguments[:3] != ["cellcli", "-xml", "-e"]:
            sys.stdout.buffer.write(fault("SIM-011: 仅允许 cellcli -xml -e 的只读查询"))
            return
        sys.stdout.buffer.write(render(arguments[3]))
        return
    if sys.argv[1:]:
        raise SystemExit("不支持此模拟入口参数。")
    # 专用容器内创建无登录密码的采集账户；密码认证和交互 Shell 均由 sshd 禁用。
    try:
        account = pwd.getpwnam("cellmonitor")
    except KeyError:
        subprocess.run(["useradd", "--no-create-home", "--uid", "54322", "--gid", "oinstall",
                        "--shell", "/bin/sh", "cellmonitor"], check=True)
    else:
        if account.pw_uid != 54322 or account.pw_shell != "/bin/sh":
            raise SystemExit("已有采集账户身份不符，停止。")
    subprocess.run(["usermod", "--password", "*", "cellmonitor"], check=True)
    home = Path("/home/cellmonitor")
    home.mkdir(mode=0o750, exist_ok=True)
    os.chown(home, 54322, 54321)
    Path("/run/sshd").mkdir(mode=0o755, exist_ok=True)
    os.execv("/usr/sbin/sshd", ["/usr/sbin/sshd", "-D", "-e", "-f", "/sim/sshd_config"])


if __name__ == "__main__":
    main()
