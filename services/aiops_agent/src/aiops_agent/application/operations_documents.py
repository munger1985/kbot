"""运维业务与本 App 规范文档之间的用例端口；不模拟旧 KC/Bundle API。"""

from dataclasses import replace
from hashlib import sha256
import json
from uuid import UUID

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.contracts import SourceDescriptor
from knowledge_algorithms.parsing.events import ParseEventType
from knowledge_algorithms.contracts.content import ContentNode
from platform_core.identity import uuid7
from ..domain.knowledge_ingestion import OWNER
from ..domain.knowledge_retrieval import RetrievalAccess
from .errors import AIOpsApplicationError


class OperationsDocuments:
    def __init__(self, *, uow_factory, ingestion, projection, retrieval, parse_profile, pipeline):
        self.uow_factory, self.ingestion, self.projection = uow_factory, ingestion, projection
        self.retrieval, self.parse_profile, self.pipeline = retrieval, parse_profile, pipeline

    @staticmethod
    def access(context, *, actor_id=None, run_ref=None, spec_hash=None):
        require(context.app_id == OWNER, "运维知识只能由 AIOps 自身访问")
        tenant = UUID(str(context.tenant_id)) if context.tenant_id else None
        domain = UUID(str(context.domain_id)) if tenant else int(context.domain_id)
        return RetrievalAccess(domain, tenant, str(actor_id or getattr(context, "principal_id", None) or getattr(context, "asserted_user_id", None) or context.client_id),
                               run_ref=run_ref, spec_hash=spec_hash)

    def configuration(self, models):
        """模型不能只写配置而没有对应表示空间；禁止猜测向量维度或模型能力。"""
        from knowledge_algorithms.contracts.identity import fingerprint
        from knowledge_algorithms.contracts.wire import encode
        specs = self.pipeline.profile.representations
        requested = {str(value) for role, value in models.items()
                     if role in {"embedding", "visual_embedding"}}
        configured = {str(spec.model_ref) for spec in specs}
        if requested != configured:
            raise AIOpsApplicationError(
                code="AIOPS_KNOWLEDGE_PROFILE_MISMATCH",
                message="所选向量模型与 AIOps 服务端表示空间不匹配，请由管理员配置模型、维度、版本和检索方向后重试",
                status_code=409,
            )
        return {"models": {key: str(value) for key, value in models.items()},
                "profile_ref": fingerprint(encode(self.pipeline.profile))}

    async def stage(self, *, file_name, media_type, body, expected_hash=None, document_id=None, version_id=None):
        require(type(body) is bytes and 0 < len(body) <= self.parse_profile.max_input_bytes,
                "运维知识文件大小超出解析预算")
        require(file_name and len(file_name) <= 512 and "/" not in file_name and "\\" not in file_name,
                "运维知识文件名无效")
        digest = sha256(body).hexdigest()
        require(expected_hash is None or expected_hash == digest, "运维知识上传正文哈希不一致")
        stored = await self.ingestion.store.put_bytes(body, area="sources",
                                                     max_bytes=self.parse_profile.max_input_bytes)
        document_id, version_id = document_id or uuid7(), version_id or uuid7()
        return {"document_id": document_id, "document_version_id": version_id,
            "source": SourceDescriptor(OWNER, str(version_id), digest, media_type, len(body), stored.handle_ref),
            "file_name": file_name}

    async def register(self, uow, access, *, staged, asset_id, asset_version_id):
        job_id = await uow.knowledge_ingestion.register(access, document_id=staged["document_id"],
            document_version_id=staged["document_version_id"], source=staged["source"],
            profile=self.parse_profile, business_asset_ref="ops-asset:" + str(asset_id),
            business_version_ref=str(asset_version_id), title=staged["file_name"], now=self.ingestion.clock())
        return {"document_id": staged["document_id"], "document_version_id": staged["document_version_id"],
            "ingestion_job_id": job_id}

    async def retry(self, access, asset_version_id):
        async with self.uow_factory() as uow:
            version = await uow.operations_knowledge.get_version(domain_id=access.domain_id,
                asset_version_id=asset_version_id, for_update=True)
            require(version is not None and version.status in {"PROCESSING", "TAGGING"}, "业务版本不能重试")
            relations = await uow.operations_knowledge.get_version_relations(asset_version_id=asset_version_id)
            for index in relations["indexes"]:
                await uow.knowledge_documents.retry_document(access, index.document_version_id, now=self.ingestion.clock())
            await uow.commit()

    async def inspect(self, access, version_id):
        async with self.uow_factory() as uow:
            return await uow.knowledge_documents.inspect(access, UUID(str(version_id)))

    async def process(self, access, version_id):
        record = await self.inspect(access, version_id)
        require(record["status"] != "WITHDRAWN", "已撤回文档不能继续处理")
        if record["processing_status"] == "PARSED" and record["generation_id"]:
            return record
        subject = replace(access, subject_ref=record["subject_ref"])
        lease = await self.ingestion.get_job(subject, UUID(record["ingestion_job_id"]))
        require(self.pipeline is not None, "运维知识流水线尚未装配")
        if (lease.status == "PARSING" and lease.lease_until > self.ingestion.clock()
            and lease.lease_owner != self.pipeline.worker_id):
            return record
        result = await self.pipeline.process(subject, UUID(record["ingestion_job_id"]))
        return {**await self.inspect(access, version_id), "generation_id": result["generation_id"]}

    async def excerpts(self, access, version_ids, *, max_chars=100000):
        require(0 < max_chars <= 1000000 and len(version_ids) <= 128, "运维知识预览预算无效")
        excerpts, used = [], 0
        for version_id in version_ids:
            record = await self.inspect(access, version_id)
            require(record["processing_status"] == "PARSED", "文档解析尚未完成")
            subject = replace(access, subject_ref=record["subject_ref"])
            lease = await self.ingestion.get_job(subject, UUID(record["ingestion_job_id"]))
            validator = await self.ingestion._validator(lease)
            require(validator.completed, "文档没有通过解析终态验证")
            async for event, _ in self.projection._events(lease):
                if event.event_type != ParseEventType.NODE_BATCH:
                    continue
                for raw in json.loads(event.payload_json)["objects"]:
                    node = decode(ContentNode, {key: value for key, value in raw.items() if key != "local_key"})
                    if not node.text:
                        continue
                    if used + len(node.text) > max_chars:
                        return {"excerpts": excerpts, "warnings": ["正文预览达到预算，未展示全部内容"]}
                    used += len(node.text)
                    excerpts.append({"document_version_id": str(version_id), "file_name": record["title"],
                        "content": node.text, "locator": {"locators": raw["locators"], "key": node.key.stable_key}})
        return {"excerpts": excerpts, "warnings": []}

    async def publish(self, uow, access, indexes):
        require(indexes, "运维知识版本没有文档引用")
        for index in indexes:
            record = await uow.knowledge_documents.inspect(access, index.document_version_id)
            require(record["document_id"] == str(index.document_id)
                and record["ingestion_job_id"] == str(index.ingestion_job_id)
                and record["parse_run_id"] == str(index.parse_run_id)
                and record["generation_id"] == str(index.index_generation_id), "文档任务或索引代次已经变化")
            await uow.knowledge_retrieval.publish(access, version_id=index.document_version_id,
                generation_id=index.index_generation_id,
                expected_generation_id=UUID(record["active_generation_id"]) if record["active_generation_id"] else None,
                now=self.ingestion.clock())

    async def withdraw(self, uow, access, indexes, *, cancel=False):
        leases = []
        for index in indexes:
            record = await uow.knowledge_documents.inspect(access, index.document_version_id)
            if cancel and record["processing_status"] not in {"PARSED", "FAILED", "CANCELLED"}:
                lease = await uow.knowledge_ingestion.get_job(replace(access, subject_ref=record["subject_ref"]), index.ingestion_job_id)
                await uow.knowledge_ingestion.cancel(lease, now=self.ingestion.clock())
                leases.append(lease)
            if record["status"] != "WITHDRAWN":
                await uow.knowledge_retrieval.withdraw(access, version_id=index.document_version_id,
                    expected_revision=record["revision"], now=self.ingestion.clock())
        return leases

    async def source(self, access, version_id, *, authorize):
        record = await self.inspect(access, version_id)
        async def chunks():
            await authorize()
            digest, count = sha256(), 0
            async for block in self.ingestion.store.stream(record["source_handle"]):
                await authorize()
                count += len(block)
                require(count <= record["byte_count"], "原文件超过登记大小")
                digest.update(block)
                yield block
            require(count == record["byte_count"] and digest.hexdigest() == record["source_hash"],
                    "运维知识原文件摘要发生变化")
        return record, chunks()
