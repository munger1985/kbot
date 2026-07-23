"""Internal INDEX worker protocol, separate from Parser callbacks."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from platform_core.contracts import INTERNAL_API_V1
from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/index-tasks",
    tags=["Knowledge Core Index"],
)


class IndexClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    max_tasks: int = Field(default=1, ge=1, le=32)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class IndexRunRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    batch_size: int = Field(default=64, ge=1, le=500)


class IndexHeartbeatRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class IndexFailRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    failure_class: str = Field(min_length=1, max_length=16)
    failure_code: str = Field(min_length=1, max_length=128)
    failure_message: str | None = None


@router.post("/claim")
async def claim_index_tasks(payload: IndexClaimRequest, request: Request):
    try:
        tasks = await request.app.state.kc_index_service.claim(
            ParseTaskClaim(payload.worker_id, payload.max_tasks, payload.lease_seconds)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CLAIM", "message": str(exc)}) from exc
    return {"tasks": [task.__dict__ | {"lease_until": task.lease_until.isoformat()} for task in tasks]}


@router.post("/{job_id}/run")
async def run_index_task(job_id: UUID, payload: IndexRunRequest, request: Request):
    try:
        status = await request.app.state.kc_index_service.run_job(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint, batch_size=payload.batch_size,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "INDEX lease is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INDEX", "message": str(exc)}) from exc
    return {"job_id": job_id, "status": "SUCCEEDED", "revision_status": status}


@router.post("/{job_id}/heartbeat")
async def heartbeat_index_task(job_id: UUID, payload: IndexHeartbeatRequest, request: Request):
    try:
        lease_until = await request.app.state.kc_index_service.heartbeat(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint, lease_seconds=payload.lease_seconds,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "INDEX lease is stale"}) from exc
    return {"job_id": job_id, "lease_until": lease_until.isoformat()}


@router.post("/{job_id}/fail")
async def fail_index_task(job_id: UUID, payload: IndexFailRequest, request: Request):
    try:
        status = await request.app.state.kc_index_service.fail(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint,
            failure_class=payload.failure_class, failure_code=payload.failure_code,
            failure_message=payload.failure_message,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "INDEX lease is stale"}) from exc
    return {"job_id": job_id, "status": status}
