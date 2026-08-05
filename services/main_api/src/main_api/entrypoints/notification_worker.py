"""Main API 通知 Outbox 投影 Worker。"""

from __future__ import annotations

import asyncio
import signal

from loguru import logger

from main_api.application import NotificationDispatcher
from main_api.config import get_main_api_settings
from main_api.persistence import create_main_api_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager


async def main() -> None:
    settings = get_main_api_settings()
    config = settings.notifications
    LogManager(LogConfig(
        service="main_api",
        process="notification-worker",
        log_dir=settings.log.dir,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
    )).setup()
    runtime = create_database_runtime()
    dispatcher = NotificationDispatcher(
        uow_factory=create_main_api_uow(runtime.session_factory),
        batch_size=config.dispatcher_batch_size,
        lease_seconds=config.dispatcher_lease_seconds,
        max_attempts=config.dispatcher_max_attempts,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for event_signal in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(event_signal, stop.set)
    logger.info("通知投影 Worker 已启动")
    try:
        while not stop.is_set():
            processed = await dispatcher.dispatch_once()
            if processed:
                continue
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=config.dispatcher_poll_seconds,
                )
            except TimeoutError:
                pass
    finally:
        await runtime.close()
        logger.info("通知投影 Worker 已停止")


if __name__ == "__main__":
    asyncio.run(main())
