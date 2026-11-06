"""独立外部通知投递进程；禁止在 API lifespan 启动此循环。"""

import asyncio
import signal

from loguru import logger
from platform_core.logger import LogConfig, LogManager
from main_api.config import get_main_api_settings
from main_api.application.notification_delivery_bootstrap import create_external_notifications
from main_api.persistence import create_main_api_uow
from platform_core.database.oracle import create_database_runtime


async def main():
    settings = get_main_api_settings()
    config = settings.external_delivery
    config.validate_runtime(production=settings.is_production())
    LogManager(LogConfig(service="main_api", process="external_notification_worker",
        log_dir=settings.log.dir, level=settings.log.level,
        rotation=settings.log.rotation, retention=settings.log.retention)).setup()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    if not config.enabled:
        logger.info("外部通知未启用，进程空闲等待停止信号，不访问数据库或飞书")
        await stop.wait()
        return
    runtime = create_database_runtime()
    service = create_external_notifications(settings=settings, runtime=runtime, uow_factory=create_main_api_uow(runtime.session_factory))
    logger.info("独立外部通知投递进程已启动")
    try:
        while not stop.is_set():
            try:
                await service.dispatch_once()
            except Exception as exc:
                logger.warning("外部通知批次失败 | error_type={}", type(exc).__name__)
            try:
                await asyncio.wait_for(stop.wait(), timeout=config.poll_seconds)
            except TimeoutError:
                pass
    finally:
        await service.sender.close()
        await service.authority.client.close()
        await runtime.close()
        logger.info("独立外部通知投递进程已停止")


if __name__ == "__main__":
    asyncio.run(main())
