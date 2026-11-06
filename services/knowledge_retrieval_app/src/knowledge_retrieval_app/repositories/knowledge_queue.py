"""本 App 正式知识队列；只枚举固定自有表并从持久任务恢复可信范围。"""

from datetime import timedelta
from sqlalchemy import select, exists, or_
from knowledge_algorithms.contracts.errors import require
from ..domain.knowledge_ingestion import IngestionScope
from ..entities.knowledge_ingestion_job import KnowledgeIngestionJobEntity as Job
from ..entities.knowledge_index_generation import KnowledgeIndexGenerationEntity as Generation
from ..entities.knowledge_outbox import KnowledgeOutboxEntity as Outbox


class KnowledgeJobQueueRepository:
    def __init__(self, session, guard):
        self._session, self._guard = session, guard

    async def next(self, *, now):
        self._guard()
        rows = (await self._session.scalars(select(Job).where(
            Job.status.in_(("RECEIVED", "PARSING", "PARSED")),
            or_(Job.retry_at.is_(None), Job.retry_at <= now),
            or_(Job.status != "PARSING", Job.lease_until <= now),
            or_(Job.status != "PARSED", exists(select(Outbox.outbox_id).where(
                Outbox.document_version_id == Job.document_version_id,
                Outbox.status.in_(("PENDING", "RUNNING")),
                or_(Outbox.retry_at.is_(None), Outbox.retry_at <= now),
                or_(Outbox.lease_until.is_(None), Outbox.lease_until <= now)))),
            or_(Job.status != "PARSED", ~exists(select(Generation.index_generation_id).where(
                Generation.document_version_id == Job.document_version_id,
                Generation.parse_run_id == Job.parse_run_id, Generation.status == "READY"))))
            .order_by(Job.retry_at, Job.ingestion_job_id).limit(64))).all()
        for row in rows:
            if row.context_json.get("pipeline_status") == "READY":
                continue
            if row.status == "PARSING" and row.lease_until > now:
                continue
            scope = IngestionScope(int(row.domain_id), getattr(row, "tenant_id", None),
                                   row.context_json["subject_ref"])
            outbox_id = None
            if row.status == "PARSED":
                outbox_id = (await self._session.scalars(select(Outbox.outbox_id).where(
                    Outbox.document_version_id == row.document_version_id,
                    Outbox.status.in_(("PENDING", "RUNNING")),
                    or_(Outbox.retry_at.is_(None), Outbox.retry_at <= now),
                    or_(Outbox.lease_until.is_(None), Outbox.lease_until <= now))
                    .order_by(Outbox.created_at).limit(1))).first()
                if outbox_id is None:
                    continue
            return scope, row.ingestion_job_id, outbox_id
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
                Outbox.document_version_id == row.document_version_id).with_for_update())).one_or_none()
            if (outbox is None or outbox.lease_owner != worker_id
                    or outbox.status != ("PENDING" if error_code else "DELIVERED")):
                return
        state = dict(row.context_json)
        if error_code:
            count = int(state.get("pipeline_failures", 0)) + 1
            state.update(pipeline_failures=count, pipeline_error=error_code, pipeline_status="RETRY")
            row.retry_at = now + timedelta(seconds=min(300, 2 ** min(count, 8)))
            if count >= 5:
                row.status, row.error_code = "FAILED", error_code
        else:
            require(row.status == "PARSED" and generation_id, "不能确认未完成的知识流水线")
            state.update(pipeline_status="READY", generation_id=str(generation_id))
            state.pop("pipeline_error", None)
        row.context_json = state
