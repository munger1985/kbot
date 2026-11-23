"""正式摄取队列驱动；不自动审核，首次索引成功后按库审批自动发布。"""

import asyncio
from time import monotonic
from loguru import logger
from knowledge_algorithms.contracts.errors import AlgorithmError
from ..domain.knowledge_ingestion import utcnow


class KnowledgeBusinessWorker:
    def __init__(self, *, uow_factory, pipeline, worker_id, poll_seconds):
        self.uow_factory, self.pipeline, self.worker_id = uow_factory, pipeline, worker_id
        self.poll_seconds, self.stopped = poll_seconds, asyncio.Event()

    def stop(self):
        self.stopped.set()

    async def recover_ready_rebuilds(self):
        """恢复进程启动前已交付、但尚未切换的批次。"""
        after = None
        while not self.stopped.is_set():
            async with self.uow_factory() as uow:
                after, batches = await uow.knowledge_queue.rebuild_recovery_page(after)
            for scope, library_id in batches:
                if self.stopped.is_set():
                    return
                try:
                    async with self.uow_factory() as uow:
                        library = uow.library(scope)
                        row = await library.get(library_id, lock=True)
                        if row is not None:
                            await library.complete_rebuild(row, publish=uow.knowledge_retrieval.publish,
                                                           now=self.pipeline.ingestion.clock())
                        await uow.commit()
                except Exception as exc:
                    code = exc.code.value if isinstance(exc, AlgorithmError) else "KNOWLEDGE_PIPELINE_FAILED"
                    logger.warning("恢复索引批次失败：library_id={} code={}", library_id, code)
            if after is None:
                return

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
            stage = exc.stage if isinstance(exc, AlgorithmError) else ("index" if outbox_id else "parse")
            logger.warning("知识流水线失败：job_id={} code={} stage={}", job_id, code, stage)
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
        await self.recover_ready_rebuilds()
        next_cleanup = 0
        while not self.stopped.is_set():
            if monotonic() >= next_cleanup:
                next_cleanup = monotonic() + 3600
                try:
                    async with self.uow_factory() as uow:
                        removed = await uow.purge_search_history(now=utcnow())
                        await uow.commit()
                    if removed == 500:
                        next_cleanup = monotonic() + 60
                except Exception:
                    logger.warning("检索历史过期清理未完成，将在下一轮重试")
            handled = await self.process_once()
            if not handled:
                try:
                    await asyncio.wait_for(self.stopped.wait(), self.poll_seconds)
                except TimeoutError:
                    pass
