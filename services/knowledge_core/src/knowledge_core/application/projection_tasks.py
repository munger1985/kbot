"""KC Projection Worker 的统一任务抢占用例。"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from knowledge_core.domain.parse_tasks import ParseTaskClaim, claim_job
from knowledge_core.persistence import KnowledgeCoreUnitOfWork


PROJECTION_JOB_TYPES = ("COLLECTION_PURGE", "INDEX", "PROFILE")


@dataclass(frozen=True)
class ClaimedProjectionTask:
    job_id: UUID
    job_type: str
    worker_id: str
    input_fingerprint: str
    lease_until: datetime
    collection_id: UUID
    parse_view_id: UUID | None = None


class KnowledgeCoreProjectionTaskService:
    """一次查询抢占 Purge、Index 或 Profile 任务。"""

    def __init__(
        self,
        *,
        uow_factory: Callable[[], KnowledgeCoreUnitOfWork],
    ):
        self._uow_factory = uow_factory

    async def claim(
        self,
        claim: ParseTaskClaim,
    ) -> list[ClaimedProjectionTask]:
        claim.validate()
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work 未初始化")
            jobs = await uow.jobs.claim_candidates_by_types(
                job_types=PROJECTION_JOB_TYPES,
                now=now,
                limit=claim.max_tasks,
            )
            tasks: list[ClaimedProjectionTask] = []
            for job in jobs:
                if (
                    job.job_type == "INDEX"
                    and job.parse_view_id is None
                    and (job.payload_json or {}).get("target") != "DISCOVERY"
                ):
                    continue
                lease_until = claim_job(job, claim, now)
                tasks.append(ClaimedProjectionTask(
                    job_id=job.ingestion_job_id,
                    job_type=job.job_type,
                    worker_id=claim.worker_id,
                    input_fingerprint=job.input_fingerprint,
                    lease_until=lease_until,
                    collection_id=job.collection_id,
                    parse_view_id=job.parse_view_id,
                ))
            if tasks:
                await uow.session.flush()
                await uow.commit()
            return tasks
