"""输出任务状态、租约围栏与结果资格；不包含持久化操作。"""

from datetime import datetime
from typing import Protocol
from uuid import UUID


class OutputResult(Protocol):
    report_id: UUID | None

    def model_dump(self, *, mode: str) -> dict: ...


class OutputJobConflictError(RuntimeError):
    """任务状态或执行围栏冲突，由应用边界映射为公开错误。"""


def job_conflict(message: str = "输出任务状态或执行租约已变化"):
    return OutputJobConflictError(message)


def can_claim(job, *, now: datetime) -> bool:
    if job.status not in {"QUEUED", "RUNNING"} or job.available_at > now:
        return False
    return job.lease_until is None or job.lease_until <= now


def start_attempt(job, *, now: datetime, owner: str, token: UUID, until: datetime) -> None:
    if not owner or until <= now or not can_claim(job, now=now):
        raise job_conflict()
    job.status = "RUNNING"
    if job.stage == "ACCEPTED":
        job.stage = "VALIDATING"
    job.attempt_count = int(job.attempt_count) + 1
    job.started_at = job.started_at or now
    job.lease_owner, job.lease_token, job.lease_until = owner, token, until
    job.heartbeat_at, job.updated_at = now, now


def require_lease(job, *, now: datetime, owner: str, token: UUID) -> None:
    if (job.status != "RUNNING" or job.lease_owner != owner
        or job.lease_token != token or job.lease_until is None or job.lease_until <= now):
        raise job_conflict()


def release_lease(job) -> None:
    job.lease_owner = job.lease_token = job.lease_until = None


def complete_job(job, *, result: OutputResult, now: datetime, owner: str, token: UUID) -> None:
    require_lease(job, now=now, owner=owner, token=token)
    if (job.job_type == "REPORT_GENERATION") != (result.report_id is not None):
        raise job_conflict("输出任务与产物类型不一致")
    job.result_json = result.model_dump(mode="json")
    job.status, job.stage = "SUCCEEDED", "COMPLETED"
    job.completed_at = job.updated_at = now
    job.retryable = False
    job.error_code = job.error_message = None
    release_lease(job)


def fail_job(job, *, code: str, message: str, retryable: bool, now: datetime,
    owner: str, token: UUID) -> None:
    require_lease(job, now=now, owner=owner, token=token)
    job.failure_stage = job.stage
    job.status, job.stage = "FAILED", "FAILED"
    job.error_code, job.error_message = code, message
    job.retryable = retryable
    job.completed_at = job.updated_at = now
    release_lease(job)


def retry_job(job, *, expected_version: int, now: datetime) -> None:
    if (job.status != "FAILED" or not job.retryable
        or int(job.row_version) != expected_version):
        raise job_conflict("当前任务不允许重试，请刷新任务状态")
    job.status = "QUEUED"
    job.stage = job.failure_stage or "VALIDATING"
    job.available_at = job.updated_at = now
    job.completed_at = None
    job.retryable = False
    job.error_code = job.error_message = None
    release_lease(job)
