"""故障进程的生命周期、超时与自动恢复。"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from pathlib import Path

from aiops_simulator.config import SimulatorConfig
from aiops_simulator.faults.base import (
    FAULT_TTL_SECONDS,
    FaultAdapter,
    normalize_database_name,
    validate_fault_type,
)
from aiops_simulator.faults.state import utc_now, write_state


LOGGER = logging.getLogger("aiops_simulator")


def _database_config(config: SimulatorConfig, public_name: str):
    engine = normalize_database_name(public_name)
    database = next((item for item in config.databases if item.engine == engine), None)
    if database is None or not database.enabled:
        raise ValueError(f"数据库{public_name}没有在日常流量配置中启用")
    return database


def _create_adapter(
    config: SimulatorConfig,
    public_name: str,
    run_id: str,
    fault_type: str,
    stop_event: asyncio.Event,
) -> FaultAdapter:
    database = _database_config(config, public_name)
    if database.engine == "oracle":
        from aiops_simulator.faults.oracle import OracleFaultAdapter

        adapter_type = OracleFaultAdapter
    elif database.engine == "postgresql":
        from aiops_simulator.faults.postgresql import PostgreSQLFaultAdapter

        adapter_type = PostgreSQLFaultAdapter
    else:
        from aiops_simulator.faults.mysql import MySQLFaultAdapter

        adapter_type = MySQLFaultAdapter
    return adapter_type(database, config, run_id, fault_type, stop_event)


async def run_fault(
    config: SimulatorConfig,
    database_name: str,
    fault_type: str,
    state_path: Path,
) -> None:
    """运行一个有硬时限且可由信号提前恢复的数据库故障。"""

    normalize_database_name(database_name)
    validate_fault_type(fault_type)
    run_id = str(uuid.uuid4())
    stop_event = asyncio.Event()
    adapter = _create_adapter(
        config, database_name, run_id, fault_type, stop_event
    )
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, stop_event.set)

    base_state = {
        "database": database_name,
        "fault_type": fault_type,
        "run_id": run_id,
        "pid": os.getpid(),
        "ttl_seconds": FAULT_TTL_SECONDS,
        "started_at": utc_now(),
    }
    write_state(state_path, **base_state, status="starting")
    task = asyncio.create_task(adapter.run())
    outcome = "failed"
    try:
        ready_task = asyncio.create_task(adapter.ready_event.wait())
        startup_stop = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(
            {task, ready_task, startup_stop},
            timeout=30,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if startup_stop in done:
            outcome = "stopped"
            ready_task.cancel()
            await asyncio.gather(ready_task, return_exceptions=True)
            return
        if task in done:
            ready_task.cancel()
            startup_stop.cancel()
            await asyncio.gather(
                ready_task, startup_stop, return_exceptions=True
            )
            await task
            raise RuntimeError("故障任务在进入运行状态前意外结束")
        if ready_task not in done:
            ready_task.cancel()
            startup_stop.cancel()
            await asyncio.gather(
                ready_task, startup_stop, return_exceptions=True
            )
            raise TimeoutError("故障在30秒内未进入运行状态")
        startup_stop.cancel()
        await asyncio.gather(startup_stop, return_exceptions=True)
        write_state(state_path, **base_state, status="active")
        LOGGER.info(
            "%s故障已启动，数据库=%s，运行标识=%s，最长持续=%d秒",
            fault_type,
            database_name,
            run_id,
            FAULT_TTL_SECONDS,
        )

        stop_waiter = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait(
            {task, stop_waiter},
            timeout=FAULT_TTL_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            await task
            raise RuntimeError("故障任务意外结束")
        if stop_waiter in done:
            outcome = "stopped"
        else:
            outcome = "expired"
            stop_event.set()
        stop_waiter.cancel()
        await asyncio.gather(stop_waiter, return_exceptions=True)
    finally:
        stop_event.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.close()
        write_state(
            state_path,
            **base_state,
            status=outcome,
            finished_at=utc_now(),
        )
        LOGGER.info(
            "故障已恢复，数据库=%s，故障类型=%s，结果=%s",
            database_name,
            fault_type,
            outcome,
        )
