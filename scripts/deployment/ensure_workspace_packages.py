"""启动前确保内部包与当前仓库源码完全一致。"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

try:
    from .workspace_fingerprint import ROOT, compare_workspace_packages
except ImportError:
    from workspace_fingerprint import ROOT, compare_workspace_packages


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("development", "production"),
        required=True,
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _install(mode: str) -> None:
    command = ["bash", "scripts/deployment/install_workspace.sh"]
    if mode == "production":
        command.append("--production")
    environment = dict(os.environ)
    environment["KBOT_PYTHON"] = sys.executable
    environment["KBOT_SKIP_LOCAL_ENV_INIT"] = "1"
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
    )


def _verify_in_fresh_interpreter() -> bool:
    """用新解释器重新加载 editable .pth 和刚安装的 Wheel。"""
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/workspace_fingerprint.py",
        ],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    """在跨进程锁内检查并按需更新内部包。"""
    arguments = _arguments()
    lock_path = ROOT / "var" / "run" / "workspace-package-install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        mismatches = compare_workspace_packages()
        if not mismatches:
            print("KBot 启动包预检通过：内部包与源码一致")
            return 0
        print("KBot 启动包预检发现不一致：")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        if arguments.check_only:
            return 1
        if os.getenv("KBOT_AUTO_INSTALL_PACKAGES", "true").lower() not in {
            "1", "true", "yes", "on"
        }:
            print("自动更新已关闭，拒绝使用旧内部包启动")
            return 1
        print(f"正在以 {arguments.mode} 模式更新 KBot 内部包……")
        try:
            _install(arguments.mode)
        except subprocess.CalledProcessError as exc:
            print(f"KBot 内部包自动更新失败：exit_code={exc.returncode}")
            return 1
        if not _verify_in_fresh_interpreter():
            print("KBot 内部包更新后仍与源码不一致")
            return 1
        print("KBot 内部包自动更新完成，允许启动服务")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
