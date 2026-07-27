"""由内部 HTTP 协议和 Worker 共享的纯解析任务租约规则。"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


class ParseLeaseError(ValueError):
    """A callback does not own a current lease for the declared input."""


@dataclass(frozen=True)
class ParseTaskClaim:
    worker_id: str
    max_tasks: int
    lease_seconds: int = 120

    def validate(self) -> None:
        if not self.worker_id.strip():
            raise ValueError("worker_id is required")
        if not 1 <= self.max_tasks <= 32:
            raise ValueError("max_tasks must be between 1 and 32")
        if not 30 <= self.lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600")


def claim_job(job, claim: ParseTaskClaim, now: datetime | None = None) -> datetime:
    """Apply a lease transition after the repository has locked the candidate row."""
    claim.validate()
    now = now or datetime.now(timezone.utc)
    waiting = (
        job.job_status in {"PENDING", "RETRY_WAIT"}
        and job.available_at <= now
    )
    expired = (
        job.job_status == "RUNNING"
        and job.lease_until is not None
        and job.lease_until <= now
    )
    if not (waiting or expired):
        raise ParseLeaseError("JOB_NOT_CLAIMABLE")
    job.job_status = "RUNNING"
    job.lease_owner = claim.worker_id
    job.lease_until = now + timedelta(seconds=claim.lease_seconds)
    job.heartbeat_at = now
    job.started_at = job.started_at or now
    job.attempt_count += 1
    job.row_version += 1
    return job.lease_until


def verify_lease(job, *, worker_id: str, input_fingerprint: str, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    if job.job_status != "RUNNING" or job.lease_owner != worker_id:
        raise ParseLeaseError("JOB_LEASE_INVALID")
    if job.input_fingerprint != input_fingerprint:
        raise ParseLeaseError("JOB_STALE")
    if job.lease_until is None or job.lease_until <= now:
        raise ParseLeaseError("JOB_LEASE_INVALID")
