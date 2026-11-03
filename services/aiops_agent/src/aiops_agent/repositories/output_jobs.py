"""输出任务仓储：作用域查询、来源串行锁与执行租约，不拥有提交权。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select

from aiops_agent.domain.output_jobs import can_claim, start_attempt
from platform_core.identity import uuid7
from aiops_agent.entities import (
    OpsConversationEntity, OpsRunEntity, ReportEntity,
    OpsOutputJobEntity, OpsOutputJobAttemptEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class OutputJobRepository(AIOpsRepository):
    async def get_for_worker(self, *, job_id: UUID):
        """仅供内部消息核验，公开读取必须使用带 Domain 和 Actor 的 get。"""
        self._check_active()
        return (await self._session.execute(select(OpsOutputJobEntity).where(
            OpsOutputJobEntity.job_id == job_id,
        ))).scalar_one_or_none()

    async def add(self, job: OpsOutputJobEntity):
        return await self._add(job)

    async def lock_source(self, *, domain_id, source_kind: str, source_id: UUID):
        """锁定已有来源根，串行受理同源任务，无业务唯一索引。"""
        self._check_active()
        entity, identity = {
            "CONVERSATION": (OpsConversationEntity, OpsConversationEntity.conversation_id),
            "RUN": (OpsRunEntity, OpsRunEntity.ops_run_id),
            "REPORT": (ReportEntity, ReportEntity.report_id),
        }[source_kind]
        statement = select(entity).where(
            entity.domain_id == domain_id, identity == source_id,
        ).with_for_update().execution_options(populate_existing=True)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_submission(self, *, domain_id, actor_id, job_type: str,
        source_kind: str, source_id: UUID, key_hash: str, input_hash: str):
        self._check_active()
        return list((await self._session.scalars(select(OpsOutputJobEntity).where(
            OpsOutputJobEntity.domain_id == domain_id,
            OpsOutputJobEntity.actor_id == actor_id,
            OpsOutputJobEntity.job_type == job_type,
            OpsOutputJobEntity.source_kind == source_kind,
            OpsOutputJobEntity.source_id == source_id,
            or_(
                OpsOutputJobEntity.idempotency_key_hash == key_hash,
                OpsOutputJobEntity.input_hash == input_hash,
            ),
        ).order_by(OpsOutputJobEntity.job_id.desc()))).all())

    async def get(self, *, domain_id, job_id: UUID, actor_id=None, lock: bool = False):
        self._check_active()
        statement = select(OpsOutputJobEntity).where(
            OpsOutputJobEntity.domain_id == domain_id,
            OpsOutputJobEntity.job_id == job_id,
        )
        if actor_id is not None:
            statement = statement.where(OpsOutputJobEntity.actor_id == actor_id)
        if lock:
            statement = statement.with_for_update().execution_options(populate_existing=True)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id, actor_id, limit: int = 50,
        job_type: str | None = None, status: str | None = None,
        conversation_id: UUID | None = None, before_id: UUID | None = None):
        self._check_active()
        statement = select(OpsOutputJobEntity).where(
            OpsOutputJobEntity.domain_id == domain_id,
            OpsOutputJobEntity.actor_id == actor_id,
        )
        for column, value in (
            (OpsOutputJobEntity.job_type, job_type),
            (OpsOutputJobEntity.status, status),
            (OpsOutputJobEntity.conversation_id, conversation_id),
        ):
            if value is not None:
                statement = statement.where(column == value)
        if before_id is not None:
            statement = statement.where(OpsOutputJobEntity.job_id < before_id)
        return list((await self._session.scalars(
            statement.order_by(OpsOutputJobEntity.job_id.desc()).limit(limit + 1)
        )).all())

    async def _next_claimable(self, *, now: datetime):
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT JOB_ID FROM KBOT_OPS_OUTPUT_JOB
                        WHERE STATUS IN ('QUEUED', 'RUNNING')
                          AND AVAILABLE_AT <= :claim_now
                          AND (LEASE_UNTIL IS NULL OR LEASE_UNTIL <= :claim_now)
                        ORDER BY AVAILABLE_AT, JOB_ID
                        FOR UPDATE OF JOB_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={"claim_now": now},
        )
        if claimed_id is None:
            return None
        return (await self._session.execute(select(OpsOutputJobEntity).where(
            OpsOutputJobEntity.job_id == claimed_id,
        ).execution_options(populate_existing=True))).scalar_one()

    async def claim(self, *, now: datetime, owner: str, token: UUID, until: datetime):
        self._check_active()
        job = await self._next_claimable(now=now)
        if job is None:
            return None
        if not can_claim(job, now=now):
            return None
        if job.lease_token is not None:
            await self.finish_attempt(
                job_id=job.job_id, token=job.lease_token, status="EXPIRED",
                stage=job.stage, now=now,
            )
        start_attempt(job, now=now, owner=owner, token=token, until=until)
        await self._add(OpsOutputJobAttemptEntity(
            attempt_id=uuid7(),
            job_id=job.job_id, attempt_no=job.attempt_count,
            lease_token=token, lease_owner=owner, status="RUNNING",
            started_at=now, stage=job.stage,
        ))
        return job

    async def finish_attempt(self, *, job_id: UUID, token: UUID, status: str,
        stage: str, now: datetime, error_code: str | None = None, error_message: str | None = None):
        self._check_active()
        attempt = (await self._session.execute(
            select(OpsOutputJobAttemptEntity).where(
                OpsOutputJobAttemptEntity.job_id == job_id,
                OpsOutputJobAttemptEntity.lease_token == token,
                OpsOutputJobAttemptEntity.status == "RUNNING",
            ).with_for_update()
        )).scalar_one_or_none()
        if attempt is not None:
            attempt.status, attempt.stage = status, stage
            attempt.completed_at = now
            attempt.error_code, attempt.error_message = error_code, error_message
        return attempt

    async def list_attempts(self, *, job_id: UUID, limit: int = 100):
        self._check_active()
        rows = list((await self._session.scalars(
            select(OpsOutputJobAttemptEntity).where(
                OpsOutputJobAttemptEntity.job_id == job_id,
            ).order_by(OpsOutputJobAttemptEntity.attempt_no.desc()).limit(limit)
        )).all())
        return list(reversed(rows))
