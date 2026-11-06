"""知识检索 App 的正式解析与索引 Worker 入口。"""

import asyncio
import signal
from platform_core.identity import uuid7
from fastapi import FastAPI
from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
from knowledge_retrieval_app.composition.knowledge_business import install_business_knowledge, business_pipeline
from knowledge_retrieval_app.config import get_knowledge_retrieval_app_settings
from knowledge_retrieval_app.persistence import create_knowledge_retrieval_app_uow
from platform_core.database.oracle import create_database_runtime
from platform_core.logger import LogConfig, LogManager


async def main():
    settings = get_knowledge_retrieval_app_settings()
    LogManager(LogConfig(service="knowledge_retrieval_app", process="knowledge_worker",
        log_dir=settings.log.dir, level=settings.log.level,
        rotation=settings.log.rotation, retention=settings.log.retention)).setup()
    runtime = create_database_runtime(settings)
    uow_factory = create_knowledge_retrieval_app_uow(runtime.session_factory)
    state = FastAPI()
    caller = settings.api.service_name.removesuffix("-api") + "-worker"
    install_business_knowledge(state, settings=settings, uow_factory=uow_factory, caller_service=caller)
    worker_id = caller + ":" + str(uuid7())
    worker = KnowledgeBusinessWorker(uow_factory=uow_factory,
        pipeline=business_pipeline(state, settings=settings, worker_id=worker_id), worker_id=worker_id,
        poll_seconds=settings.knowledge.poll_seconds)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, worker.stop)
    try:
        await worker.run_forever()
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
