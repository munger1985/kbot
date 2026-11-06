"""App 摄取用例：外部读取在事务外，持久提交成功后才 ACK。"""

from dataclasses import asdict
import json
from uuid import UUID

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.parsing.events import ParseStreamValidator
from knowledge_algorithms.parsing.wire import event_from_wire, event_to_wire
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.contracts import ParseProfile
from knowledge_retrieval_app.domain.knowledge_ingestion import BINDING, utcnow
from knowledge_retrieval_app.ports.knowledge_ingestion import AuthorizedSourceReader


class KnowledgeIngestionService:
    def __init__(self, *, uow_factory, object_store, compute, clock=utcnow,
                 page_limit=100, max_artifact_bytes=67108864, max_history_events=100000):
        require(1 <= page_limit <= 1000 and max_artifact_bytes > 0 and max_history_events > 0, "摄取资源预算无效")
        self.uow_factory, self.store, self.compute = uow_factory, object_store, compute
        self.clock, self.page_limit = clock, page_limit
        self.max_artifact_bytes, self.max_history_events = max_artifact_bytes, max_history_events

    async def register(self, scope, *, document_id, document_version_id, source, profile,
                       source_blocks, business_asset_ref, business_version_ref, title):
        BINDING.check_owner(source.owner_binding_ref)
        require(decode(ParseProfile, encode(profile)).profile_hash == profile.profile_hash,
                "解析 Profile 数值必须采用稳定的标准线格式")
        copied = await self.store.put(source_blocks, area="sources", expected_sha256=source.sha256,
            expected_size=source.byte_count, mime_type=source.mime_type, max_bytes=profile.max_input_bytes)
        require(source.source_handle == copied.handle_ref, "源描述必须绑定本 App 实际存储的受控句柄")
        async with self.uow_factory() as uow:
            job_id = await uow.knowledge_ingestion.register(scope, document_id=document_id,
                document_version_id=document_version_id, source=source, profile=profile,
                business_asset_ref=business_asset_ref, business_version_ref=business_version_ref,
                title=title, now=self.clock())
            await uow.commit()
        return job_id

    async def claim(self, scope, job_id, *, worker_id, lease_seconds):
        async with self.uow_factory() as uow:
            lease = await uow.knowledge_ingestion.claim(scope, job_id, worker_id=worker_id,
                now=self.clock(), lease_seconds=lease_seconds)
            await uow.commit()
        await self.ensure_compute(lease)
        return await self.get_job(scope, job_id)

    async def ensure_compute(self, lease):
        """提交响应或本地绑定丢失时，使用同一冻结身份恢复计算回执。"""
        lease.require_live(self.clock())
        compute_ref = await self.compute.submit(lease)
        async with self.uow_factory() as uow:
            await uow.knowledge_ingestion.bind_compute(lease, compute_ref, now=self.clock())
            await uow.commit()
        return compute_ref

    async def get_job(self, scope, job_id):
        async with self.uow_factory() as uow:
            return await uow.knowledge_ingestion.get_job(scope, job_id)

    async def renew(self, lease, *, lease_seconds):
        async with self.uow_factory() as uow:
            result = await uow.knowledge_ingestion.renew(lease, now=self.clock(), lease_seconds=lease_seconds)
            await uow.commit()
        await self.ensure_compute(result)
        return result

    async def _validator(self, lease):
        profile = lease.profile
        validator = ParseStreamValidator(lease.identity, max_events=min(self.max_history_events, profile.max_objects * 4 + 4),
            max_objects=profile.max_objects, max_payload_bytes=profile.max_event_bytes)
        after = 0
        while after < lease.sequence:
            async with self.uow_factory() as uow:
                page = await uow.knowledge_ingestion.batches(lease, after_sequence=after, limit=self.page_limit)
            require(page and page[-1].sequence <= lease.sequence, "接收历史缺页或并发改变")
            for receipt in page:
                encoded = await self.store.read(receipt.payload_ref, max_bytes=profile.max_event_bytes * 2 + 8192)
                event = event_from_wire(json.loads(encoded))
                require(event.event_digest == receipt.event_digest and event.sequence == receipt.sequence,
                        "持久批次摘要或顺序失效")
                validator.accept(event)
                after = event.sequence
        require(validator.event_count == lease.sequence, "接收历史不连续")
        return validator

    async def poll(self, scope, job_id):
        lease = await self.get_job(scope, job_id)
        if lease.status == "PARSING" and lease.compute_ref is None:
            await self.ensure_compute(lease)
            lease = await self.get_job(scope, job_id)
        if lease.status == "CANCELLED":
            return {"sequence": lease.sequence, "status": lease.status}
        if lease.sequence:
            # 重启或 ACK 丢失时重送已提交恢复点，而不是重新执行解析。
            await self._acknowledge(lease, sequence=lease.sequence, event_digest=lease.last_event_digest)
        if lease.status in {"PARSED", "FAILED", "CANCELLED"}:
            return {"sequence": lease.sequence, "status": lease.status}
        lease.require_live(self.clock())
        validator = await self._validator(lease)
        events = await self.compute.events(lease, after_sequence=lease.sequence, limit=self.page_limit)
        require(len(events) <= self.page_limit, "计算端返回超过事件页预算")
        if not events:
            return {"sequence": lease.sequence, "status": "PARSING"}
        refs, artifacts = [], []
        for event in events:
            changed = validator.accept(event)
            artifact_map = {}
            if changed:
                for artifact in event.artifact_handles:
                    if artifact.expires_at <= int(self.clock().timestamp()):
                        raise AlgorithmError(ErrorCode.ARTIFACT_EXPIRED, "计算制品已过期，不能确认或发布", stage="receive")
                    copied = await self.store.put(self.compute.artifact(lease, artifact.handle_ref),
                        area="parsed", expected_sha256=artifact.sha256, expected_size=artifact.byte_count,
                        mime_type=artifact.mime_type, max_bytes=self.max_artifact_bytes)
                    artifact_map[artifact.handle_ref] = asdict(copied)
            encoded = json.dumps(event_to_wire(event), ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
            stored = await self.store.put_bytes(encoded, max_bytes=lease.profile.max_event_bytes * 2 + 8192)
            refs.append(stored.handle_ref)
            artifacts.append(artifact_map)
        async with self.uow_factory() as uow:
            sequence = await uow.knowledge_ingestion.receive(lease, expected_sequence=lease.sequence,
                events=events, payload_refs=refs, artifact_maps=artifacts, now=self.clock())
            await uow.commit()
        last = next(event for event in reversed(events) if event.sequence == sequence)
        await self._acknowledge(lease, sequence=sequence, event_digest=last.event_digest)
        current = await self.get_job(scope, job_id)
        return {"sequence": sequence, "status": current.status}

    async def _acknowledge(self, lease, *, sequence, event_digest):
        await self.compute.acknowledge(lease, sequence=sequence, event_digest=event_digest)
        # 只有收到计算端成功回执才记录；写回失败时下一轮安全重送 ACK。
        async with self.uow_factory() as uow:
            await uow.knowledge_ingestion.mark_acknowledged(lease, sequence=sequence,
                event_digest=event_digest, now=self.clock())
            await uow.commit()

    async def cancel(self, scope, job_id):
        lease = await self.get_job(scope, job_id)
        # 先撤销 App 栅栏；计算端取消失败也不能让迟到结果被持久接收。
        async with self.uow_factory() as uow:
            await uow.knowledge_ingestion.cancel(lease, now=self.clock())
            await uow.commit()
        await self.compute.cancel(lease)

    async def source_reader(self, scope, job_id, *, identity):
        lease = await self.get_job(scope, job_id)
        require(lease.identity == identity, "源读取委托与当前任务不一致")
        lease.require_live(self.clock())
        async def authorize():
            current = await self.get_job(scope, job_id)
            require(current.identity == identity, "源读取委托已失效")
            current.require_live(self.clock())
        return AuthorizedSourceReader(lease.descriptor, self.store, authorize)
