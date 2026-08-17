"""KM Asset Slack Inbox/Outbox 独立 Worker。"""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import aiohttp
from loguru import logger

from km_asset_app.application import SlackDispatchService
from km_asset_app.config import get_km_asset_settings
from km_asset_app.persistence import create_km_asset_uow
from platform_clients import (
    AgentRuntimeClient,
    KmAssetClient,
    KnowledgeCoreClient,
)
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager


async def main() -> None:
    settings = get_km_asset_settings()
    worker_config = settings.slack_worker
    slack_config = settings.integrations.slack
    LogManager(
        LogConfig(
            service="km_asset_app",
            process="slack_worker",
            log_dir=settings.log.dir,
            level=settings.log.level,
            rotation=settings.log.rotation,
            retention=settings.log.retention,
        )
    ).setup()
    if slack_config.enabled:
        for workspace in slack_config.workspaces:
            workspace.require_signing_secret()
            workspace.require_bot_token()
    debug_file: Path | None = None
    if slack_config.debug.callback_payload_log_enabled:
        debug_file = (
            Path(settings.log.dir)
            / "km_asset_app"
            / "slack_callback_debug.log"
        )
        logger.warning("Slack Callback 完整报文调试日志已启用")
    if slack_config.debug.slack_reply_dump_enabled:
        logger.warning(
            "Slack 原始回复报文调试文件已启用：{}",
            slack_config.debug.slack_reply_dump_dir,
        )
    database = create_database_runtime()
    timeout = max(
        settings.agent_runtime.timeout_seconds,
        settings.km_asset_api.timeout_seconds,
        settings.knowledge_core.timeout_seconds,
        slack_config.external_callback.timeout_seconds,
        30,
    )
    http_session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)
    if not slack_config.enabled:
        logger.info("Slack 集成未启用，Slack Worker 进入待机状态")
        await stop_event.wait()
        await http_session.close()
        await database.close()
        return
    try:
        dispatcher = SlackDispatchService(
            uow_factory=create_km_asset_uow(database.session_factory),
            agent_client=AgentRuntimeClient(
                base_url=settings.agent_runtime.base_url,
                caller_service=worker_config.service_name,
                audience=settings.agent_runtime.audience,
                timeout_seconds=settings.agent_runtime.timeout_seconds,
                session=http_session,
            ),
            km_asset_client=KmAssetClient(
                base_url=settings.km_asset_api.base_url,
                caller_service=worker_config.service_name,
                audience=settings.km_asset_api.audience,
                timeout_seconds=settings.km_asset_api.timeout_seconds,
                session=http_session,
            ),
            knowledge_core_client=KnowledgeCoreClient(
                base_url=settings.knowledge_core.base_url,
                caller_service=worker_config.service_name,
                audience=settings.knowledge_core.audience,
                timeout_seconds=settings.knowledge_core.timeout_seconds,
                session=http_session,
            ),
            slack_config=slack_config,
            worker_id=worker_config.worker_id,
            http_session=http_session,
            callback_debug_log_path=debug_file,
        )
        logger.info("Slack Worker 已启动：worker_id={}", worker_config.worker_id)
        while not stop_event.is_set():
            worked = await dispatcher.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=slack_config.outbox_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass
    finally:
        await http_session.close()
        await database.close()
        logger.info("Slack Worker 已停止")


if __name__ == "__main__":
    asyncio.run(main())
