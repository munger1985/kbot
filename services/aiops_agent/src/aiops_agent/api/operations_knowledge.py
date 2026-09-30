"""AIOps 运维知识库内部 API。"""

import json
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from aiops_agent.contracts.knowledge import (
    KnowledgeReviewCommand,
    KnowledgeSearchRequest,
    ManualUploadMetadata,
)
from platform_core.contracts import AuthContext


router = APIRouter(
    prefix="/internal/v1/aiops/operations-knowledge",
    tags=["AIOps Operations Knowledge"],
)


def _scope(request: Request, context: AuthContext) -> tuple[int, str]:
    require_service_scope(request, "aiops.manage")
    if context.domain_id is None or int(context.domain_id) < 1:
        from fastapi import HTTPException
        raise HTTPException(403, {"code": "AIOPS_DOMAIN_CONTEXT_REQUIRED"})
    return int(context.domain_id), context.asserted_user_id or context.client_id


@router.get("/overview")
async def overview(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.overview(
        domain_id=domain_id
    )


@router.get("/assets")
async def list_assets(
    request: Request,
    asset_kind: str | None = None,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_assets(
        domain_id=domain_id, asset_kind=asset_kind, status=status, limit=limit
    )}


@router.get("/assets/{asset_id}")
async def get_asset(
    asset_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_asset(
        domain_id=domain_id, asset_id=asset_id
    )


@router.get("/versions/{asset_version_id}")
async def get_version(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.get_version(
        domain_id=domain_id, asset_version_id=asset_version_id
    )


@router.get("/versions/{asset_version_id}/source")
async def stream_source(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    result = await request.app.state.operations_knowledge_service.stream_source(
        domain_id=domain_id, asset_version_id=asset_version_id,
        context=context, range_header=request.headers.get("Range"),
    )
    headers = {
        key: value for key, value in result.headers.items()
        if key in {"content-length", "content-range", "content-disposition", "accept-ranges"}
    }
    return StreamingResponse(
        result.body,
        status_code=result.status_code,
        media_type=result.headers.get("content-type", "application/octet-stream"),
        headers=headers,
    )


@router.post("/manuals:ingest", status_code=202)
async def ingest_manual(
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    raw_metadata = unquote(request.headers.get("X-Upload-Metadata", ""))
    metadata = ManualUploadMetadata.model_validate(json.loads(raw_metadata))
    body = await request.body()
    return await request.app.state.operations_knowledge_service.ingest_manual(
        domain_id=domain_id,
        context=context,
        file_name=unquote(request.headers.get("X-File-Name", "manual.bin")),
        media_type=request.headers.get("Content-Type", "application/octet-stream"),
        body=body,
        metadata=metadata,
        idempotency_key=request.headers.get("Idempotency-Key", ""),
        declared_sha256=request.headers.get("X-Content-SHA256"),
    )


@router.post("/assets/{asset_id}/versions:ingest", status_code=202)
async def ingest_manual_version(
    asset_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    raw_metadata = unquote(request.headers.get("X-Upload-Metadata", ""))
    metadata = ManualUploadMetadata.model_validate(json.loads(raw_metadata))
    try:
        expected_asset_row_version = int(
            request.headers.get("X-Expected-Asset-Row-Version", "")
        )
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(
            422,
            {"code": "AIOPS_KNOWLEDGE_VERSION_CONFLICT", "message": "缺少有效的资产版本号"},
        ) from exc
    body = await request.body()
    return await request.app.state.operations_knowledge_service.ingest_manual_version(
        domain_id=domain_id,
        asset_id=asset_id,
        expected_asset_row_version=expected_asset_row_version,
        context=context,
        file_name=unquote(request.headers.get("X-File-Name", "manual.bin")),
        media_type=request.headers.get("Content-Type", "application/octet-stream"),
        body=body,
        metadata=metadata,
        idempotency_key=request.headers.get("Idempotency-Key", ""),
        declared_sha256=request.headers.get("X-Content-SHA256"),
    )


@router.post("/reports/{report_id}:extract-case", status_code=202)
async def extract_case(
    report_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.extract_case(
        domain_id=domain_id, report_id=report_id, context=context
    )


@router.post("/versions/{asset_version_id}:reconcile")
async def reconcile_version(
    asset_version_id: UUID,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.reconcile_version(
        domain_id=domain_id, asset_version_id=asset_version_id, context=context
    )


async def _review(
    *, decision: str, asset_version_id: UUID, payload: KnowledgeReviewCommand,
    request: Request, context: AuthContext,
):
    domain_id, _ = _scope(request, context)
    return await request.app.state.operations_knowledge_service.review(
        domain_id=domain_id,
        asset_version_id=asset_version_id,
        decision=decision,
        expected_row_version=payload.expected_row_version,
        comment=payload.comment,
        context=context,
    )


@router.post("/versions/{asset_version_id}:publish")
async def publish_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                          request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="PUBLISH", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:reject")
async def reject_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                         request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="REJECT", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:retire")
async def retire_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                         request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="RETIRE", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.post("/versions/{asset_version_id}:retry")
async def retry_version(asset_version_id: UUID, payload: KnowledgeReviewCommand,
                        request: Request, context: AuthContext = Depends(get_aiops_auth_context)):
    return await _review(decision="RETRY", asset_version_id=asset_version_id,
                         payload=payload, request=request, context=context)


@router.get("/reviews")
async def list_reviews(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, _ = _scope(request, context)
    return {"items": await request.app.state.operations_knowledge_service.list_reviews(
        domain_id=domain_id, limit=limit
    )}


@router.post(":search")
async def search(
    payload: KnowledgeSearchRequest,
    request: Request,
    context: AuthContext = Depends(get_aiops_auth_context),
):
    domain_id, actor = _scope(request, context)
    return await request.app.state.operations_knowledge_service.search(
        domain_id=domain_id, agent_id=actor, request=payload, context=context
    )
