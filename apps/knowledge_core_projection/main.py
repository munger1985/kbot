"""Knowledge Core indexing/profile worker process.

This process is deliberately separate from the KC HTTP API and Parser Worker:
it owns only INDEX and PROFILE leases and communicates through authenticated
internal endpoints.
"""

import asyncio
import os
import signal

from loguru import logger

from platform_core.config.settings import get_app_config, get_knowledge_core_config
from knowledge_core.workers.projection.client import KcIndexProfileClient
from knowledge_core.workers.projection.worker import KcIndexProfileWorker


async def main() -> None:
    app_config = get_app_config()
    config = get_knowledge_core_config()
    worker_id = os.getenv("KBOT_KC_WORKER_ID", "kbot-kc-index-profile-v2")
    worker = KcIndexProfileWorker(
        client=KcIndexProfileClient(base_url=f"http://127.0.0.1:{config.service_port}", timeout_seconds=600),
        worker_id=worker_id,
        lease_seconds=int(os.getenv("KBOT_KC_WORKER_LEASE_SECONDS", "600")),
        poll_interval=float(os.getenv("KBOT_KC_WORKER_POLL_INTERVAL", "2")),
        index_batch_size=int(os.getenv("KBOT_KC_INDEX_BATCH_SIZE", "64")),
    )
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, worker.stop)
    logger.info(
        "Starting KC INDEX/PROFILE worker {} -> {}:{} (app_id={})",
        worker_id, config.service_host, config.service_port, app_config.app_id,
    )
    await worker.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
