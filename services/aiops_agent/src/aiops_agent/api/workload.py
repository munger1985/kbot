"""MySQL/PostgreSQL工作负载快照、报告和活动采样Internal API。"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request, Response, status

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.errors import validation_failed
from aiops_agent.application.workload import WorkloadService
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ActivitySampleBatch,
    ReportArtifactView,
    ReportWindow,
    WorkloadDiffRequest,
    WorkloadSnapshotCreate,
)
from platform_core.contracts.aiops.workload import PGBADGER_MAX_UPLOAD_BYTES


router = APIRouter(
    prefix="/internal/v1/aiops/workload",
    tags=["AIOps Workload"],
)
artifact_router = APIRouter(
    prefix="/internal/v1/aiops",
    tags=["AIOps Artifacts"],
)
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key")]


def get_service(request: Request) -> WorkloadService:
    return request.app.state.workload_service


Service = Annotated[WorkloadService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _scope(context: AuthContext) -> tuple[int, str]:
    if context.domain_id is None or int(context.domain_id) < 1:
        raise RuntimeError("工作负载请求缺少有效Domain")
    actor_id = context.asserted_user_id or context.client_id
    return int(context.domain_id), actor_id


async def _bounded_upload_body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise validation_failed("pgBadger上传Content-Length无效") from exc
        if declared_size < 0:
            raise validation_failed("pgBadger上传Content-Length无效")
        if declared_size > PGBADGER_MAX_UPLOAD_BYTES:
            raise validation_failed("pgBadger上传文件超过20MiB")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > PGBADGER_MAX_UPLOAD_BYTES:
            raise validation_failed("pgBadger上传文件超过20MiB")
    return bytes(body)


@router.post("/workload-snapshots", status_code=status.HTTP_201_CREATED)
async def record_workload_snapshot(
    body: WorkloadSnapshotCreate,
    request: Request,
    service: Service,
    context: Auth,
) -> dict[str, str]:
    require_service_scope(request, "aiops.task")
    domain_id, _ = _scope(context)
    snapshot_id = await service.record_snapshot(
        domain_id=domain_id,
        request=body,
    )
    return {"workload_snapshot_id": str(snapshot_id)}


@router.post("/activity-samples", status_code=status.HTTP_201_CREATED)
async def record_activity_samples(
    body: ActivitySampleBatch,
    request: Request,
    service: Service,
    context: Auth,
) -> dict[str, object]:
    require_service_scope(request, "aiops.task")
    domain_id, _ = _scope(context)
    sample_ids = await service.record_activity_samples(
        domain_id=domain_id,
        target_id=body.target_id,
        sampled_at=body.sampled_at,
        database_type=str(body.database_type),
        instance_identity=body.instance_identity,
        samples=tuple(item.model_dump(mode="json") for item in body.samples),
        retention_days=body.retention_days,
    )
    return {
        "sample_count": len(sample_ids),
        "activity_sample_ids": [str(value) for value in sample_ids],
    }


@router.post(
    "/reports/workload",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_workload_report(
    body: ReportWindow,
    request: Request,
    service: Service,
    context: Auth,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    require_service_scope(request, "aiops.run")
    domain_id, actor_id = _scope(context)
    return await service.create_workload_report(
        domain_id=domain_id,
        actor_id=actor_id,
        target_id=body.target_id,
        period_start=body.period_start,
        period_end=body.period_end,
        idempotency_key=idempotency_key,
        trace_id=context.trace_id,
    )


@router.post(
    "/reports/workload-diff",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_workload_diff_report(
    body: WorkloadDiffRequest,
    request: Request,
    service: Service,
    context: Auth,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    require_service_scope(request, "aiops.run")
    domain_id, actor_id = _scope(context)
    return await service.create_workload_diff_report(
        domain_id=domain_id,
        actor_id=actor_id,
        target_id=body.target_id,
        baseline_start=body.baseline_start,
        baseline_end=body.baseline_end,
        after_start=body.after_start,
        after_end=body.after_end,
        idempotency_key=idempotency_key,
        trace_id=context.trace_id,
    )


@router.post(
    "/reports/activity",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_activity_report(
    body: ReportWindow,
    request: Request,
    service: Service,
    context: Auth,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    require_service_scope(request, "aiops.run")
    domain_id, actor_id = _scope(context)
    return await service.create_activity_report(
        domain_id=domain_id,
        actor_id=actor_id,
        target_id=body.target_id,
        period_start=body.period_start,
        period_end=body.period_end,
        idempotency_key=idempotency_key,
        trace_id=context.trace_id,
    )


@router.post(
    "/postgresql/pgbadger-artifacts",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def import_postgresql_pgbadger_artifact(
    request: Request,
    service: Service,
    context: Auth,
    idempotency_key: IdempotencyKey,
    target_id: UUID = Query(),
    period_start: datetime = Query(),
    period_end: datetime = Query(),
    file_name: str = Header(alias="X-File-Name"),
) -> ReportArtifactView:
    require_service_scope(request, "aiops.run")
    domain_id, actor_id = _scope(context)
    return await service.import_pgbadger_artifact(
        domain_id=domain_id,
        actor_id=actor_id,
        target_id=target_id,
        period_start=period_start,
        period_end=period_end,
        file_name=unquote(file_name)[:256],
        content_type=request.headers.get(
            "content-type", "application/octet-stream"
        ).split(";", 1)[0],
        body=await _bounded_upload_body(request),
        idempotency_key=idempotency_key,
        trace_id=context.trace_id,
    )


@artifact_router.get("/artifacts/{artifact_id}/content")
async def download_report_artifact(
    artifact_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> Response:
    require_service_scope(request, "aiops.run")
    domain_id, _ = _scope(context)
    content, media_type, file_name = await service.get_report_artifact_content(
        domain_id=domain_id,
        artifact_id=artifact_id,
    )
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{file_name}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
