"""独立 本地监控 Agent 运行入口；安装须显式执行，不自动覆盖未完成的卷。"""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path("/opt/oem-remote")
HOME = ROOT / "agent/agent_24.1.0.0.0"
READY = ROOT / ".configured"
STOPPING = False


def command(*arguments):
    return ["setpriv", "--reuid=oracle", "--regid=oinstall", "--init-groups", "--", *map(str, arguments)]


def status():
    if not READY.is_file():
        return False
    try:
        result = subprocess.run(command(HOME / "bin/emctl", "status", "agent"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        return (result.returncode == 0
                and b"Agent is Running and Ready" in result.stdout
                and b"Collections enabled" in result.stdout
                and re.search(rb"Heartbeat Status\s*:\s*Ok", result.stdout) is not None)
    except (OSError, subprocess.TimeoutExpired):
        return False


def stop(*_arguments):
    global STOPPING
    STOPPING = True


def main():
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for directory in (ROOT, Path("/opt/remote-state")):
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, 54321, 54321)
        directory.chmod(0o750)
    print("独立 本地监控 Agent 容器就绪；等待显式安装或启动已有配置，不自动安装。", flush=True)
    while not STOPPING and not READY.is_file():
        time.sleep(2)
    if STOPPING:
        return
    receipt = json.loads(READY.read_text())
    if receipt.get("local_agent") is not True:
        raise RuntimeError("缺少匹配的 本地监控 Agent 配置凭证。")
    info = (HOME / "sbin/nmo").stat()
    if info.st_uid != 0 or not info.st_mode & 0o4000:
        raise RuntimeError("未完成原厂 Agent root 配置，保留现场。")
    environment = dict(os.environ, HOME="/home/oracle", USER="oracle", LOGNAME="oracle")
    if not status():
        subprocess.run(command(HOME / "bin/emctl", "start", "agent"), env=environment, check=True)
    while not STOPPING:
        time.sleep(2)
    subprocess.run(command(HOME / "bin/emctl", "stop", "agent"), env=environment, check=True)


def boot():
    """实际运行标准 SysVinit；不自己写 runlevel/utmp，不增加宿主机权限。"""
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    Path("/run").mkdir(exist_ok=True)
    # 初始化正常的空记账文件；BOOT_TIME/RUN_LVL 记录只能由真实 init 写入。
    Path("/run/utmp").touch(mode=0o644, exist_ok=True)
    Path("/var/log/wtmp").touch(mode=0o644, exist_ok=True)
    process = subprocess.Popen(["/usr/local/sbin/sysvinit-init", "--init", "3"],
                               env=dict(os.environ, CONSOLE="/dev/null"))
    print("已启动真实 SysVinit 3.14；由它管理 runlevel 和 Agent 服务，不使用模拟的 who/utmp。", flush=True)
    try:
        while not STOPPING:
            if process.poll() is not None:
                raise RuntimeError("真实 init 进程已退出。")
            time.sleep(2)
    finally:
        if READY.is_file():
            subprocess.run(command(HOME / "bin/emctl", "stop", "agent"), timeout=90, check=False)
        # 这是容器内部的非 PID1 子进程；Agent 已先正常停止，随后回收 init。
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


if __name__ == "__main__":
    if sys.argv[1:] == ["--health"]:
        raise SystemExit(0 if status() else 1)
    if sys.argv[1:] == ["--service"]:
        main()
    else:
        boot()
