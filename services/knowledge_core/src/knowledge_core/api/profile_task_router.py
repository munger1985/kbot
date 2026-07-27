"""Internal PROFILE worker protocol."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from platform_core.contracts import INTERNAL_API_V1
from knowledge_core.domain.parse_tasks import ParseLeaseError

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/profile-tasks",
    tags=["Knowledge Core Profile"],
)


class ProfileRunRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    input_fingerprint: str = Field(min_length=64, max_length=64)


class ProfileHeartbeatRequest(ProfileRunRequest):
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class ProfileFailRequest(ProfileRunRequest):
    failure_class: str = Field(min_length=1, max_length=16)
    failure_code: str = Field(min_length=1, max_length=128)
    failure_message: str | None = None


@router.post("/{job_id}/run")
async def run_profile_task(job_id: UUID, payload: ProfileRunRequest, request: Request):
    try:
        count = await request.app.state.kc_profile_service.run_job(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "PROFILE lease is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PROFILE", "message": str(exc)}) from exc
    return {"job_id": job_id, "status": "STAGED", "profile_count": count}


@router.post("/{job_id}/heartbeat")
async def heartbeat_profile_task(job_id: UUID, payload: ProfileHeartbeatRequest, request: Request):
    try:
        lease_until = await request.app.state.kc_profile_service.heartbeat(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint, lease_seconds=payload.lease_seconds,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "PROFILE lease is stale"}) from exc
    return {"job_id": job_id, "lease_until": lease_until.isoformat()}


@router.post("/{job_id}/fail")
async def fail_profile_task(job_id: UUID, payload: ProfileFailRequest, request: Request):
    try:
        result = await request.app.state.kc_profile_service.fail(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint,
            failure_class=payload.failure_class, failure_code=payload.failure_code,
            failure_message=payload.failure_message,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "PROFILE lease is stale"}) from exc
    return {"job_id": job_id, "status": result}
