"""Knowledge Core indexing/profile worker process.

This process is deliberately separate from the KC HTTP API and Parser Worker:
it owns only INDEX and PROFILE leases and communicates through authenticated
internal endpoints.
"""

import asyncio
import os
import signal

from loguru import logger

from knowledge_core.config import get_knowledge_core_settings
from knowledge_core.adapters.oracle_job_wakeup import (
    OracleDbmsAlertListener,
)
from knowledge_core.ports.job_wakeup import PROJECTION_WAKEUP_CHANNEL
from knowledge_core.workers.job_wait import AdaptiveJobWait
from knowledge_core.workers.projection.client import KcIndexProfileClient
from knowledge_core.workers.projection.worker import KcIndexProfileWorker
from platform_core.logger import LogConfig, LogManager


async def main() -> None:
    settings = get_knowledge_core_settings()
    config = settings.projection
    LogManager(
        LogConfig(
            service="knowledge_core",
            process="projection",
            log_dir=settings.log.dir,
            level=settings.log.level,
            rotation=settings.log.rotation,
            retention=settings.log.retention,
        )
    ).setup()
    job_wakeup = settings.job_wakeup
    listener = (
        OracleDbmsAlertListener(
            settings=settings,
            channel=PROJECTION_WAKEUP_CHANNEL,
        )
        if job_wakeup.mode == "DBMS_ALERT"
        else None
    )
    worker = KcIndexProfileWorker(
        client=KcIndexProfileClient(
            base_url=settings.knowledge_core.base_url,
            timeout_seconds=settings.knowledge_core.timeout_seconds,
            caller_service=config.service_name,
            audience=settings.knowledge_core.audience,
        ),
        worker_id=config.worker_id,
        lease_seconds=config.lease_seconds,
        job_wait=AdaptiveJobWait(
            listener=listener,
            notification_timeout_seconds=(
                job_wakeup.notification_timeout_seconds
            ),
            fallback_min_seconds=job_wakeup.fallback_min_seconds,
            fallback_max_seconds=job_wakeup.fallback_max_seconds,
            fallback_multiplier=job_wakeup.fallback_multiplier,
            jitter_ratio=job_wakeup.jitter_ratio,
        ),
        index_batch_size=config.index_batch_size,
    )
    loop = asyncio.get_running_loop()
    worker_task = asyncio.create_task(
        worker.run_forever(),
        name="kc-projection-worker",
    )

    def stop_worker() -> None:
        worker.stop()
        worker_task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_worker)
    logger.info(
        "正在启动 KC INDEX/PROFILE Worker {} -> {}",
        config.worker_id,
        settings.knowledge_core.base_url,
    )
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
