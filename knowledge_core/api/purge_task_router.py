"""Internal lease protocol for asynchronous Collection purge."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from platform_core.contracts import INTERNAL_API_V1
from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim

router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/purge-tasks",
    tags=["Knowledge Core Internal"],
)


class PurgeClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    max_tasks: int = Field(default=1, ge=1, le=32)
    lease_seconds: int = Field(default=600, ge=30, le=3600)


class RunRequest(BaseModel):
    worker_id: str
    input_fingerprint: str


class LeaseRequest(RunRequest):
    lease_seconds: int = Field(default=600, ge=30, le=3600)


@router.post("/claim")
async def claim(payload: PurgeClaimRequest, request: Request):
    try:
        tasks = await request.app.state.kc_purge_service.claim(ParseTaskClaim(**payload.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CLAIM", "message": str(exc)}) from exc
    return {"tasks": tasks}


@router.post("/{job_id}/run")
async def run(job_id: UUID, payload: RunRequest, request: Request):
    try:
        return await request.app.state.kc_purge_service.run(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "purge lease is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PURGE", "message": str(exc)}) from exc


@router.post("/{job_id}/heartbeat")
async def heartbeat(job_id: UUID, payload: LeaseRequest, request: Request):
    try:
        lease_until = await request.app.state.kc_purge_service.heartbeat(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
            lease_seconds=payload.lease_seconds,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "purge lease is stale"}) from exc
    return {"job_id": job_id, "lease_until": lease_until}


@router.post("/{job_id}/fail")
async def fail(job_id: UUID, payload: RunRequest, request: Request):
    try:
        result = await request.app.state.kc_purge_service.fail(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
            failure_code="WORKER_RUN_FAILED", message="purge worker failed",
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "purge lease is stale"}) from exc
    return {"job_id": job_id, "status": result}
