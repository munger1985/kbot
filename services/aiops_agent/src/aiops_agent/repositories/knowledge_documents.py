"""AIOps 文档任务与业务资产的精确映射；只在本 App 的存储边界访问实体。"""

from uuid import UUID
from sqlalchemy import select
from ..entities.knowledge_configuration import OperationsKnowledgeConfigurationEntity
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import fingerprint
from .knowledge_ingestion import KnowledgeIngestionRepository


class OperationsDocumentRepository(KnowledgeIngestionRepository):
    async def inspect(self, access, version_id):
        repos = self._repos(access)
        version = await repos["knowledge_document_version"].get(version_id)
        require(version is not None, "文档版本不属于当前 AIOps 空间")
        document = await repos["knowledge_document"].get(version.document_id)
        jobs = repos["knowledge_ingestion_job"]
        rows = (await self._session.scalars(jobs.statement().where(
            jobs.entity.document_version_id == version_id))).all()
        require(document is not None and len(rows) == 1, "文档来源或任务身份断裂")
        job = rows[0]
        generation_repo = repos["knowledge_index_generation"]
        generations = (await self._session.scalars(generation_repo.statement().where(
            generation_repo.entity.document_version_id == version_id,
            generation_repo.entity.parse_run_id == job.parse_run_id,
            generation_repo.entity.status == "READY"
        ).order_by(generation_repo.entity.created_at.desc(),
                   generation_repo.entity.index_generation_id.desc()).limit(1))).all()
        generation = generations[0] if generations else None
        return {"document_id": str(document.document_id), "document_version_id": str(version_id),
            "ingestion_job_id": str(job.ingestion_job_id), "parse_run_id": str(job.parse_run_id),
            "subject_ref": job.context_json["subject_ref"], "source_hash": version.source_sha256,
            "source_handle": version.source_object_ref, "mime_type": version.mime_type,
            "byte_count": int(version.byte_count), "title": document.title,
            "business_asset_ref": document.business_asset_ref,
            "business_version_ref": version.business_version_ref,
            "status": version.status, "processing_status": job.status,
            "error_code": job.error_code, "revision": int(version.row_version),
            "generation_id": str(generation.index_generation_id) if generation else None,
            "active_generation_id": str(version.active_generation_id) if version.active_generation_id else None,
            "allowed_view_hash": version.allowed_view_hash}

    async def create_query_ticket(self, access, *, ticket_id, agent_id, target_id, request, snapshots, expires_at, filters=None, parent_turn_id=None, execution_spec=None):
        require(snapshots and len(snapshots) <= 500, "知识查询范围为空或超过预算")
        snapshots = [{key: value for key, value in row.items() if key != "created_at"} for row in snapshots]
        values = {"parent_turn_id": str(parent_turn_id) if parent_turn_id else None, "agent_id": str(agent_id), "target_id": str(target_id) if target_id else None,
            "request": request.model_dump(mode="json"), "snapshots": snapshots, "filters": filters or {},
            "execution_spec": execution_spec,
            "run_ref": getattr(access, "run_ref", None), "spec_hash": getattr(access, "spec_hash", None)}
        digest = fingerprint(values)
        await self._repos(access)["knowledge_scope_snapshot"].add(dict(
            scope_snapshot_id=ticket_id, subject_ref=access.subject_ref, scope_hash=digest,
            authorization_revision=digest, allowed_versions_hash=fingerprint(snapshots),
            policy_json=values, status="QUERY_TICKET", expires_at=expires_at))
        return "ops-query:" + str(ticket_id)

    async def child_binding(self, access, identity, *, now):
        """在锁定父轮次后读回原始冻结规格；不允许重新绑定或续租旧查询。"""
        row = await self._repos(access)["knowledge_scope_snapshot"].get(identity)
        if row is None:
            return None
        require(row.status == "QUERY_TICKET" and row.subject_ref == access.subject_ref
            and row.expires_at > now and fingerprint(row.policy_json) == row.scope_hash,
            "原知识子执行委托已经过期或被篡改")
        require(row.policy_json.get("execution_spec"), "查询不是可重放的知识子执行")
        return row.policy_json

    async def query_ticket(self, access, identity, *, now, allow_expired=False):
        row = await self._repos(access)["knowledge_scope_snapshot"].get(identity)
        require(row is not None and row.status == "QUERY_TICKET"
            and row.subject_ref == access.subject_ref and (allow_expired or row.expires_at > now),
            "知识查询委托已失效或不属于当前主体")
        value = row.policy_json
        require(value.get("run_ref") == getattr(access, "run_ref", None)
            and value.get("spec_hash") == getattr(access, "spec_hash", None)
            and fingerprint(value) == row.scope_hash
            and fingerprint(value["snapshots"]) == row.allowed_versions_hash,
            "知识查询委托的运行身份或摘要不一致")
        return value
    async def case_configuration(self, access, *, lock=False):
        self._guard()
        entity = OperationsKnowledgeConfigurationEntity
        statement = select(entity).where(entity.domain_id == access.domain_id)
        if hasattr(entity, "tenant_id"):
            statement = statement.where(entity.tenant_id == access.tenant_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def configure_cases(self, access, *, models, profile_ref):
        self._guard()
        row = await self.case_configuration(access, lock=True)
        if row is None:
            fields = {"domain_id": access.domain_id}
            if hasattr(OperationsKnowledgeConfigurationEntity, "tenant_id"):
                fields["tenant_id"] = access.tenant_id
            row = OperationsKnowledgeConfigurationEntity(**fields)
            self._session.add(row)
        row.models_json, row.profile_ref, row.updated_by = models, profile_ref, access.subject_ref
        await self._session.flush()
        return row

    async def retry_document(self, access, version_id, *, now):
        """只重试未发布失败任务；新解析代次保留旧批次并使旧租约永久失效。"""
        repos = self._repos(access)
        version = await repos["knowledge_document_version"].get(version_id, lock=True)
        require(version is not None and version.status not in {"PUBLISHED", "WITHDRAWN", "CANCELLED"}
                and version.active_generation_id is None, "发布、撤回或取消的正文不能原地重试")
        jobs = repos["knowledge_ingestion_job"]
        rows = (await self._session.scalars(jobs.statement().where(
            jobs.entity.document_version_id == version_id).with_for_update())).all()
        require(len(rows) == 1, "文档任务身份不唯一")
        job = rows[0]
        if job.status == "PARSED":
            return
        require(job.status in {"FAILED", "RECEIVED"} or (job.lease_until is not None and job.lease_until <= now),
                "运行中的有效任务不能被重试请求抢占")
        if job.status == "RECEIVED":
            return
        previous = await repos["knowledge_parse_run"].get(job.parse_run_id, lock=True)
        attempt, fence = int(job.attempt) + 1, int(job.fence) + 1
        run = await repos["knowledge_parse_run"].add(dict(document_version_id=version_id,
            source_sha256=version.source_sha256, profile_hash=previous.profile_hash,
            parser_revision="PENDING", model_snapshot_json={}, attempt=attempt, fence=fence,
            status="RECEIVED", profile_json=previous.profile_json, checkpoint_json={}))
        job.parse_run_id, job.attempt, job.fence = run.parse_run_id, attempt, fence
        job.status, job.lease_owner, job.lease_until, job.retry_at = "RECEIVED", None, now, now
        job.recovery_sequence, job.last_event_digest, job.compute_ref, job.error_code = 0, None, None, None
        version.status, version.allowed_view_hash, version.allowed_view_manifest_ref = "RECEIVED", None, None
