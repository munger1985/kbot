"""Internal Parser Worker task lease endpoints."""
from uuid import UUID
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts import INTERNAL_API_V1
from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim
from knowledge_core.application.parse_tasks import EvidenceInput


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/parse-tasks",
    tags=["Knowledge Core Parser"],
)


class ClaimRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    max_tasks: int = Field(default=1, ge=1, le=32)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class HeartbeatRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=256)
    input_fingerprint: str = Field(min_length=64, max_length=64)
    lease_seconds: int = Field(default=120, ge=30, le=3600)


class EvidenceItemRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_key: str
    evidence_type: str
    ordinal: int
    fragment_index: int = Field(ge=0)
    content_text: str
    source_spans: list[dict[str, Any]] = Field(min_length=1)
    locator_schema_version: str = Field(min_length=1, max_length=32)
    locator: dict[str, Any]
    provenance: dict[str, Any]
    parent_evidence_key: str | None = None
    source_item_ref: str | None = None
    heading_path: list[str] | None = None
    section_key: str | None = None
    hierarchy_depth: int | None = None
    heading_level: int | None = None
    payload_descriptor: dict[str, Any] | None = None
    page_start: int | None = None
    page_end: int | None = None
    language_code: str | None = None
    token_count: int | None = None
    quality_score: float | None = None


class EvidenceBatchRequest(BaseModel):
    worker_id: str
    input_fingerprint: str
    items: list[EvidenceItemRequest]


class ArtifactUploadRequest(BaseModel):
    worker_id: str
    input_fingerprint: str
    sha256: str = Field(min_length=64, max_length=64)
    schema_name: str = Field(min_length=1, max_length=128)
    generator: str = Field(min_length=1, max_length=256)
    payload: Any


class CompleteRequest(BaseModel):
    worker_id: str
    input_fingerprint: str
    artifact_manifest: dict[str, Any]
    output_fingerprint: str = Field(min_length=64, max_length=64)
    quality_score: float | None = None
    quality_report: dict[str, Any]


class FailRequest(BaseModel):
    worker_id: str
    input_fingerprint: str
    failure_class: str
    failure_code: str
    failure_message: str | None = None
    artifact_manifest: dict[str, Any] | None = None


@router.post("/claim")
async def claim_parse_tasks(payload: ClaimRequest, request: Request):
    try:
        tasks = await request.app.state.kc_parse_task_service.claim(
            ParseTaskClaim(payload.worker_id, payload.max_tasks, payload.lease_seconds)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CLAIM", "message": str(exc)}) from exc
    return {"tasks": [{
        "job_id": task.job_id,
        "lease_owner": task.lease_owner,
        "lease_until": task.lease_until.isoformat(),
        "input_fingerprint": task.input_fingerprint,
        "document_version_id": task.document_version_id,
        "parse_view_id": task.parse_view_id,
        "source_read_url": (
            str(request.base_url).rstrip("/")
            + f"{INTERNAL_API_V1}/knowledge/parse-tasks/{task.job_id}/source"
        ),
        "detected_mime_type": task.detected_mime_type,
        "view_kind": task.view_kind,
        "parse_config_fingerprint": task.parse_config_fingerprint,
        "policy_snapshot": task.policy_snapshot,
    } for task in tasks]}


@router.get("/{job_id}/source")
async def read_parse_source(
    job_id: UUID, request: Request,
    worker_id: str = Query(min_length=1, max_length=256),
    input_fingerprint: str = Query(min_length=64, max_length=64),
):
    try:
        path, mime_type = await request.app.state.kc_parse_task_service.source_descriptor(
            job_id=job_id, worker_id=worker_id, input_fingerprint=input_fingerprint,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser source lease is stale"}) from exc
    return FileResponse(path, media_type=mime_type, filename=f"document-version-{job_id}")


@router.post("/{job_id}/heartbeat")
async def heartbeat_parse_task(job_id: UUID, payload: HeartbeatRequest, request: Request):
    try:
        lease_until = await request.app.state.kc_parse_task_service.heartbeat(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint, lease_seconds=payload.lease_seconds,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser lease is no longer valid"}) from exc
    return {"job_id": job_id, "lease_until": lease_until.isoformat()}


@router.post("/{job_id}/evidence-batches")
async def submit_evidence(job_id: UUID, payload: EvidenceBatchRequest, request: Request):
    try:
        inserted = await request.app.state.kc_parse_task_service.submit_evidence(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
            items=[EvidenceInput(**item.model_dump()) for item in payload.items],
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser result is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_EVIDENCE", "message": str(exc)}) from exc
    return {"job_id": job_id, "inserted": inserted}


@router.post("/{job_id}/artifacts/{artifact_name}")
async def upload_parse_artifact(
    job_id: UUID, artifact_name: str, payload: ArtifactUploadRequest, request: Request,
):
    try:
        descriptor = await request.app.state.kc_parse_task_service.upload_artifact(
            job_id=job_id, worker_id=payload.worker_id,
            input_fingerprint=payload.input_fingerprint, artifact_name=artifact_name,
            payload=payload.payload, expected_sha256=payload.sha256,
            schema=payload.schema_name, generator=payload.generator,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser result is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_ARTIFACT", "message": str(exc)}) from exc
    return descriptor


@router.post("/{job_id}/complete")
async def complete_parse(job_id: UUID, payload: CompleteRequest, request: Request):
    try:
        count = await request.app.state.kc_parse_task_service.complete(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
            artifact_manifest=payload.artifact_manifest, output_fingerprint=payload.output_fingerprint,
            quality_score=payload.quality_score, quality_report=payload.quality_report,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser result is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PARSE_RESULT", "message": str(exc)}) from exc
    return {
        "job_id": job_id,
        "status": "PARSED",
        "evidence_count": count,
        "index_status": "PENDING",
    }


@router.post("/{job_id}/fail")
async def fail_parse(job_id: UUID, payload: FailRequest, request: Request):
    try:
        result = await request.app.state.kc_parse_task_service.fail(
            job_id=job_id, worker_id=payload.worker_id, input_fingerprint=payload.input_fingerprint,
            failure_class=payload.failure_class, failure_code=payload.failure_code,
            failure_message=payload.failure_message,
            artifact_manifest=payload.artifact_manifest,
        )
    except ParseLeaseError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Parser result is stale"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_FAILURE_RESULT", "message": str(exc)}) from exc
    return {"job_id": job_id, "status": result}
