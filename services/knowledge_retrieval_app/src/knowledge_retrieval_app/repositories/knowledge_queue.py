"""本 App 正式知识队列；只枚举固定自有表并从持久任务恢复可信范围。"""

from datetime import timedelta
from uuid import UUID
from sqlalchemy import select, or_
from knowledge_algorithms.contracts.errors import require
from ..domain.knowledge_ingestion import IngestionScope
from ..domain.knowledge_library import active_rebuild, index_rebuild_batches
from ..entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity as Job
from ..entities.knowledge_library import KnowledgeLibraryEntity as Library
from ..entities.knowledge_outbox import KnowledgeOutboxEntity as Outbox


class KnowledgeJobQueueRepository:
    def __init__(self, session, guard):
        self._session, self._guard = session, guard

    async def rebuild_recovery_page(self, after_id):
        """按键分页恢复已交付但尚未生效的候选；不扫描文件或重跑模型。"""
        self._guard()
        statement = select(Library).where(Library.status == "ACTIVE").order_by(Library.library_id).limit(50)
        if after_id is not None:
            statement = statement.where(Library.library_id > after_id)
        rows = (await self._session.scalars(statement)).all()
        batches = []
        for row in rows:
            bulk, files = index_rebuild_batches(row.configuration_json)
            if active_rebuild(bulk) or any(
                active_rebuild(batch) or batch["terminal_status"] == "PUBLISHED" and batch.get("source_published") is False
                for batch in files.values()
            ):
                scope = IngestionScope(int(row.domain_id), getattr(row, "tenant_id", None), row.created_by)
                batches.append((scope, row.library_id))
        return (rows[-1].library_id if rows else None), batches

    async def next(self, *, now):
        self._guard()
        parse_job = (await self._session.scalars(select(Job).where(
            Job.status.in_(("RECEIVED", "PARSING")),
            or_(Job.retry_at.is_(None), Job.retry_at <= now),
            or_(Job.status != "PARSING", Job.lease_until <= now))
            .order_by(Job.retry_at, Job.ingestion_job_id).limit(1))).first()
        outboxes = (await self._session.scalars(select(Outbox).where(
            Outbox.event_type.in_(("BUILD_PROJECTION", "REBUILD_PROJECTION")),
            Outbox.status.in_(("PENDING", "RUNNING")),
            or_(Outbox.retry_at.is_(None), Outbox.retry_at <= now),
            or_(Outbox.lease_until.is_(None), Outbox.lease_until <= now)
        ).order_by(Outbox.retry_at, Outbox.created_at, Outbox.outbox_id).limit(64))).all()
        for outbox in outboxes:
            if parse_job is not None and (parse_job.retry_at or parse_job.created_at) < (outbox.retry_at or outbox.created_at):
                break
            query = select(Job).where(Job.parse_run_id == UUID(outbox.idempotency_key.split(":", 1)[0]),
                Job.document_version_id == outbox.document_version_id, Job.domain_id == outbox.domain_id,
                Job.status == "PARSED")
            if hasattr(Job, "tenant_id"):
                query = query.where(Job.tenant_id == outbox.tenant_id)
            row = (await self._session.scalars(query)).one_or_none()
            if row is not None:
                scope = IngestionScope(int(row.domain_id), getattr(row, "tenant_id", None), row.context_json["subject_ref"])
                return scope, row.ingestion_job_id, outbox.outbox_id
        if parse_job is not None:
            row = parse_job
            scope = IngestionScope(int(row.domain_id), getattr(row, "tenant_id", None), row.context_json["subject_ref"])
            return scope, row.ingestion_job_id, None
        return None

    async def finish(self, scope, job_id, *, worker_id, now, generation_id=None, error_code=None, outbox_id=None):
        self._guard()
        statement = select(Job).where(Job.ingestion_job_id == job_id, Job.domain_id == scope.domain_id)
        if hasattr(Job, "tenant_id"):
            statement = statement.where(Job.tenant_id == scope.tenant_id)
        row = (await self._session.scalars(statement.with_for_update())).one_or_none()
        require(row is not None and row.context_json["subject_ref"] == scope.subject_ref, "知识队列身份不一致")
        if row.status in {"CANCELLED", "FAILED"}:
            return
        if outbox_id is None:
            if row.lease_owner != worker_id:
                return
        else:
            outbox = (await self._session.scalars(select(Outbox).where(
                Outbox.outbox_id == outbox_id,
                Outbox.event_type == "BUILD_PROJECTION",
                Outbox.document_version_id == row.document_version_id).with_for_update())).one_or_none()
            if (outbox is None or outbox.lease_owner != worker_id
                    or outbox.status != ("PENDING" if error_code else "DELIVERED")):
                return
        state = dict(row.context_json)
        if error_code:
            count = int(state.get("pipeline_failures", 0)) + 1
            state.update(pipeline_failures=count, pipeline_error=error_code, pipeline_status="RETRY")
            row.retry_at = now + timedelta(seconds=min(300, 2 ** min(count, 8)))
            if count >= 5 or (outbox_id is not None and error_code in {"RESOURCE_LIMIT", "INVALID_CONTRACT"}):
                if outbox_id is not None:
                    state["pipeline_status"] = "INDEX_FAILED"
                    outbox.status, outbox.retry_at = "FAILED", None
                    row.retry_at = None
                else:
                    row.status, row.error_code = "FAILED", error_code
        else:
            require(row.status == "PARSED" and generation_id, "不能确认未完成的知识流水线")
            state.update(pipeline_status="READY", generation_id=str(generation_id))
            state.pop("pipeline_error", None)
        row.context_json = state
