"""App 摄取到投影的可恢复驱动；发布决定仍由 App 授权/审核用例作出。"""

import asyncio
import time

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require


class KnowledgePipelineWorker:
    def __init__(self, *, ingestion, projection, builder_factory, profile, allowed_policy,
                 policy_revision, worker_id, lease_seconds=360, poll_interval_seconds=0.25):
        require(worker_id and 15 <= lease_seconds <= 600 and 0 < poll_interval_seconds <= 10,
                "知识流水线 Worker 配置无效")
        self.ingestion, self.projection, self.builder_factory = ingestion, projection, builder_factory
        self.profile, self.policy, self.policy_revision = profile, allowed_policy, policy_revision
        self.worker_id, self.lease_seconds, self.poll_interval = worker_id, lease_seconds, poll_interval_seconds

    async def process_projection(self, scope, outbox_id, job_id):
        """已完成解析的 outbox 构建使用原有租约、栅栏与失败重试。"""
        result = {}
        lease = await self.ingestion.get_job(scope, job_id)
        require(lease.status == "PARSED", "索引 outbox 不能隐式创建新的解析任务")
        async def build(version_id):
            require(version_id == lease.document_version_id, "索引任务与解析版本不一致")
            result.update(await self.process(scope, job_id))
        await self.projection.process_outbox(scope, outbox_id, worker_id=self.worker_id,
            lease_seconds=self.lease_seconds, build=build)
        return result

    async def run_forever(self, scope, *, next_job, on_failure, stop_event):
        """由 App 提供固定授权范围的队列；不枚举其他 App 或请求指定的表。"""
        from uuid import UUID
        while not stop_event.is_set():
            job_id = await next_job(scope)
            if job_id is None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    pass
                continue
            require(isinstance(job_id, UUID), "队列必须返回可信的任务 UUID")
            try:
                await self.process(scope, job_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await on_failure(scope, job_id, exc.code.value if isinstance(exc, AlgorithmError) else "PIPELINE_FAILED")

    async def process(self, scope, job_id, *, parse_only=False):
        lease = await self.ingestion.get_job(scope, job_id)
        if lease.status in {"FAILED", "CANCELLED"}:
            raise AlgorithmError(ErrorCode.PARSE_QUALITY_INSUFFICIENT, "知识任务已失败或取消", stage="pipeline")
        if lease.status != "PARSED":
            if lease.status != "PARSING" or lease.lease_until <= self.ingestion.clock():
                lease = await self.ingestion.claim(scope, job_id, worker_id=self.worker_id,
                                                   lease_seconds=self.lease_seconds)
            require(lease.lease_owner == self.worker_id, "知识任务由其他 Worker 持有")
            started = time.monotonic()
            while lease.status == "PARSING":
                if time.monotonic() - started > lease.profile.timeout_seconds + 30:
                    raise AlgorithmError(ErrorCode.TIMEOUT, "知识流水线等待计算超过预算", stage="pipeline")
                remaining = (lease.lease_until - self.ingestion.clock()).total_seconds()
                if remaining <= self.lease_seconds / 2:
                    lease = await self.ingestion.renew(lease, lease_seconds=self.lease_seconds)
                await self.ingestion.poll(scope, job_id)
                lease = await self.ingestion.get_job(scope, job_id)
                if lease.status == "PARSING":
                    await asyncio.sleep(self.poll_interval)
            require(lease.status == "PARSED", "知识任务没有完成解析，禁止建立发布索引")
        if parse_only:
            return {"status": "PARSED"}
        document = await self.projection.allowed_document(scope, job_id, policy=self.policy,
                                                          policy_revision=self.policy_revision)
        builder = self.builder_factory(scope, document)
        generation = await self.projection.prepare(scope, job_id, document=document, builder=builder, profile=self.profile)
        return {"document_version_id": str(lease.document_version_id), "generation_id": str(generation),
                "status": "READY_FOR_APP_PUBLICATION"}
