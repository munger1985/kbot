"""独立模拟程序命令入口。"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from aiops_simulator.config import load_config
from aiops_simulator.engine import SimulatorEngine
from aiops_simulator.faults import print_fault_status, run_fault
from aiops_simulator.modes import load_mode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KBot AIOps数据库流量模拟程序")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "运行日常流量"),
        ("validate", "校验日常流量配置"),
        ("prepare", "显式准备模拟程序自有表"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--config", required=True, help="权限受控的唯一INI配置文件"
        )

    fault_run = commands.add_parser("fault-run", help="运行单库故障")
    fault_run.add_argument("--config", required=True)
    fault_run.add_argument("--database", required=True)
    fault_run.add_argument("--fault-type", required=True)
    fault_run.add_argument("--state-file", required=True, type=Path)

    fault_status = commands.add_parser("fault-status", help="读取单库故障状态")
    fault_status.add_argument("--database", required=True)
    fault_status.add_argument("--state-file", required=True, type=Path)
    fault_status.add_argument("--running", action="store_true")
    return parser


async def _main_async(command: str, config_path: str) -> None:
    config = load_config(config_path)
    mode = load_mode(config.mode)
    engine = SimulatorEngine(config, mode)
    if command == "validate":
        await engine.validate()
        logging.getLogger("aiops_simulator").info("配置与三库预检全部通过")
        return
    if command == "prepare":
        await engine.prepare()
        return

    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, engine.stop)
    await engine.run()


def main() -> int:
    arguments = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if arguments.command == "fault-run":
            config = load_config(arguments.config)
            asyncio.run(
                run_fault(
                    config,
                    arguments.database,
                    arguments.fault_type,
                    arguments.state_file,
                )
            )
        elif arguments.command == "fault-status":
            print_fault_status(
                arguments.state_file,
                arguments.database,
                arguments.running,
            )
        else:
            asyncio.run(_main_async(arguments.command, arguments.config))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.getLogger("aiops_simulator").error(
            "模拟程序执行失败，错误类型=%s", type(exc).__name__
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
