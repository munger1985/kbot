"""正式摄取队列驱动；不自动审核或发布，取消后不继续写入。"""

import asyncio
from loguru import logger
from knowledge_algorithms.contracts.errors import AlgorithmError
from ..domain.knowledge_ingestion import utcnow


class KnowledgeBusinessWorker:
    def __init__(self, *, uow_factory, pipeline, worker_id, poll_seconds):
        self.uow_factory, self.pipeline, self.worker_id = uow_factory, pipeline, worker_id
        self.poll_seconds, self.stopped = poll_seconds, asyncio.Event()

    def stop(self):
        self.stopped.set()

    async def process_once(self):
        async with self.uow_factory() as uow:
            item = await uow.knowledge_queue.next(now=utcnow())
        if item is None:
            return False
        scope, job_id, outbox_id = item
        try:
            if outbox_id is None:
                await self.pipeline.process(scope, job_id, parse_only=True)
                return True
            result = await self.pipeline.process_projection(scope, outbox_id, job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = exc.code.value if isinstance(exc, AlgorithmError) else "KNOWLEDGE_PIPELINE_FAILED"
            logger.warning("知识流水线失败：job_id={} code={}", job_id, code)
            async with self.uow_factory() as uow:
                await uow.knowledge_queue.finish(scope, job_id, worker_id=self.worker_id,
                    now=utcnow(), error_code=code, outbox_id=outbox_id)
                await uow.commit()
        else:
            async with self.uow_factory() as uow:
                await uow.knowledge_queue.finish(scope, job_id, worker_id=self.worker_id,
                    now=utcnow(), generation_id=result["generation_id"], outbox_id=outbox_id)
                await uow.commit()
        return True

    async def run_forever(self):
        while not self.stopped.is_set():
            handled = await self.process_once()
            if not handled:
                try:
                    await asyncio.wait_for(self.stopped.wait(), self.poll_seconds)
                except TimeoutError:
                    pass
