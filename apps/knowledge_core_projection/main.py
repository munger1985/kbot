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
from knowledge_core.workers.projection.client import KcIndexProfileClient
from knowledge_core.workers.projection.worker import KcIndexProfileWorker


async def main() -> None:
    settings = get_knowledge_core_settings()
    config = settings.projection
    worker = KcIndexProfileWorker(
        client=KcIndexProfileClient(
            base_url=settings.knowledge_core.base_url,
            timeout_seconds=settings.knowledge_core.timeout_seconds,
            caller_service=config.service_name,
            audience=settings.knowledge_core.audience,
        ),
        worker_id=config.worker_id,
        lease_seconds=config.lease_seconds,
        poll_interval=config.poll_interval_seconds,
        index_batch_size=config.index_batch_size,
    )
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, worker.stop)
    logger.info(
        "正在启动 KC INDEX/PROFILE Worker {} -> {}（app_id={}）",
        config.worker_id,
        settings.knowledge_core.base_url,
        settings.platform.app_id,
    )
    await worker.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
