"""驱动可恢复解析与建索引；首次索引完成时按已保存的库审批自动发布。"""

import asyncio
import time
from uuid import UUID

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require


class KnowledgePipelineWorker:
    def __init__(self, *, ingestion, projection, builder_factory, profile, allowed_policy,
                 policy_revision, worker_id, lease_seconds=360, poll_interval_seconds=0.25, profile_for_version=None):
        require(worker_id and 15 <= lease_seconds <= 600 and 0 < poll_interval_seconds <= 10,
                "知识流水线 Worker 配置无效")
        self.ingestion, self.projection, self.builder_factory = ingestion, projection, builder_factory
        self.profile, self.policy, self.policy_revision = profile, allowed_policy, policy_revision
        self.profile_for_version = profile_for_version
        self.worker_id, self.lease_seconds, self.poll_interval = worker_id, lease_seconds, poll_interval_seconds

    async def process_projection(self, scope, outbox_id, job_id):
        """已完成解析的 outbox 构建使用原有租约、栅栏与失败重试。"""
        result = {}
        lease = await self.ingestion.get_job(scope, job_id)
        require(lease.status == "PARSED", "索引 outbox 不能隐式创建新的解析任务")
        async def build(claim):
            require(claim["document_version_id"] == lease.document_version_id
                    and claim["idempotency_key"].split(":", 1)[0] == lease.identity.parse_run_ref,
                    "索引任务与解析版本不一致")
            if claim["event_type"] == "REBUILD_PROJECTION":
                builder = self.builder_factory(scope, None)
                generation = await self.projection.rebuild(lease, UUID(claim["payload_ref"]), builder=builder,
                    fence={"outbox_id": outbox_id, "worker_id": self.worker_id, "attempt": claim["attempt"]})
                result["generation_id"] = str(generation)
                return
            async with self.ingestion.uow_factory() as uow:
                ready = await uow.knowledge_ingestion.ready_initial_projection(lease)
            if ready is not None:
                result["generation_id"] = str(ready)
                return
            result.update(await self.process(scope, job_id, projection_fence={
                "outbox_id": outbox_id, "worker_id": self.worker_id, "attempt": claim["attempt"],
            }))
        async def publish_completed(uow, claim):
            library = uow.library(scope)
            if claim["event_type"] == "REBUILD_PROJECTION":
                row = await library.lock_library_for_version(lease.document_version_id)
                if row is not None and row.status == "ACTIVE":
                    async def publish_rebuild():
                        await library.complete_rebuild(row, publish=uow.knowledge_retrieval.publish,
                            now=self.ingestion.clock(), batch_id=claim["idempotency_key"].rsplit(":", 1)[-1],
                            outbox_id=outbox_id, version_id=lease.document_version_id)
                    return publish_rebuild
                return
            if claim["event_type"] != "BUILD_PROJECTION":
                return
            approved, expected_generation, defer = await library.lock_approved_projection_publication(
                lease.document_version_id, job_id)
            if defer:
                return False
            if not approved:
                return
            now = self.ingestion.clock()
            await uow.knowledge_ingestion.assert_outbox_claim(scope, outbox_id, worker_id=self.worker_id,
                attempt=claim["attempt"], now=now)
            await uow.knowledge_retrieval.publish(scope, version_id=lease.document_version_id,
                generation_id=UUID(result["generation_id"]), expected_generation_id=expected_generation, now=now)

        await self.projection.process_outbox(scope, outbox_id, worker_id=self.worker_id,
            lease_seconds=self.lease_seconds, build=build, pass_claim=True, on_success=publish_completed)
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

    async def process(self, scope, job_id, *, projection_profile=None, parse_only=False, projection_fence=None):
        lease = await self.ingestion.get_job(scope, job_id)
        if lease.status in {"FAILED", "CANCELLED"}:
            if parse_only:
                return {"status": lease.status}
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
            # 解析终态及原始错误码已由摄取用例持久化，无须转成流水线合同异常。
            if parse_only and lease.status in {"FAILED", "CANCELLED"}:
                return {"status": lease.status}
            require(lease.status == "PARSED", "知识任务没有完成解析，禁止建立发布索引")
        if parse_only:
            return {"status": "PARSED"}
        document = await self.projection.allowed_document(scope, job_id, policy=self.policy,
                                                          policy_revision=self.policy_revision)
        builder = self.builder_factory(scope, document)
        if projection_profile is None:
            async with self.ingestion.uow_factory() as uow:
                projection_profile = await uow.knowledge_ingestion.replacement_projection_profile(lease)
        if projection_profile is None and self.profile_for_version is not None:
            projection_profile = await self.profile_for_version(scope, lease.document_version_id)
        generation = await self.projection.prepare(scope, job_id, document=document, builder=builder,
            profile=projection_profile or self.profile, fence=projection_fence)
        return {"document_version_id": str(lease.document_version_id), "generation_id": str(generation),
                "status": "READY_FOR_APP_PUBLICATION"}
