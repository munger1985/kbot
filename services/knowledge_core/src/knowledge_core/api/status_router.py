"""具备 Domain 边界的入库与解析状态内部端点。"""
from uuid import UUID
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import get_actor_id, require_domain_match
from knowledge_core.application.status import KnowledgeObjectNotFoundError
from knowledge_core.application.reprocessing import (
    ReprocessingConflictError,
    ReprocessingNotFoundError,
)


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}/bundles",
    tags=["Knowledge Core Status"],
)

catalog_router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}/catalog",
    tags=["Knowledge Core Catalog"],
)


class ReprocessRequest(BaseModel):
    collection_id: UUID
    document_version_id: UUID | None = None


@router.post("/{bundle_id}/revisions/{bundle_revision_id}/reprocess")
async def reprocess_revision(
    domain_id: int,
    bundle_id: UUID,
    bundle_revision_id: UUID,
    payload: ReprocessRequest,
    request: Request,
):
    require_domain_match(request, domain_id)
    try:
        result = await request.app.state.kc_reprocessing_service.reprocess(
            domain_id=domain_id,
            collection_id=payload.collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=bundle_revision_id,
            document_version_id=payload.document_version_id,
            actor_id=get_actor_id(request),
        )
    except ReprocessingNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "REPROCESS_TARGET_NOT_FOUND", "message": str(exc)},
        ) from exc
    except ReprocessingConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "REPROCESS_CONFLICT", "message": str(exc)},
        ) from exc
    return {
        "bundle_revision_id": result.bundle_revision_id,
        "generation": result.generation,
        "scheduled_file_count": result.scheduled_file_count,
    }


@router.get("/{bundle_id}")
async def get_bundle_status(domain_id: int, bundle_id: UUID, request: Request):
    require_domain_match(request, domain_id)
    try:
        result = await request.app.state.kc_status_service.get_bundle(domain_id=domain_id, bundle_id=bundle_id)
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "BUNDLE_NOT_FOUND", "message": str(exc)}) from exc
    return asdict(result)


@router.get("/{bundle_id}/revisions/{bundle_revision_id}")
async def get_revision_status(domain_id: int, bundle_id: UUID, bundle_revision_id: UUID, request: Request):
    require_domain_match(request, domain_id)
    try:
        result = await request.app.state.kc_status_service.get_revision(
            domain_id=domain_id, bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "REVISION_NOT_FOUND", "message": str(exc)}) from exc
    return asdict(result)


@router.get("/{bundle_id}/revisions/{bundle_revision_id}/members")
async def get_revision_members(domain_id: int, bundle_id: UUID, bundle_revision_id: UUID, request: Request):
    require_domain_match(request, domain_id)
    try:
        result = await request.app.state.kc_status_service.get_revision(
            domain_id=domain_id, bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
            include_members=True,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "REVISION_NOT_FOUND", "message": str(exc)}) from exc
    return asdict(result)


def _not_found(exc: KnowledgeObjectNotFoundError) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "KNOWLEDGE_OBJECT_NOT_FOUND", "message": str(exc)},
    )


@catalog_router.get("/processing")
async def list_processing(
    domain_id: int, request: Request, collection_id: UUID,
    q: str | None = None, status: str | None = None,
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
):
    require_domain_match(request, domain_id)
    try:
        return await request.app.state.kc_status_service.list_processing(
            domain_id=domain_id, collection_id=collection_id,
            query=q, status=status, page=page, page_size=page_size,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@catalog_router.get("/files")
async def list_library_files(
    domain_id: int, request: Request, collection_id: UUID,
    q: str | None = None, status: str | None = None,
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
):
    require_domain_match(request, domain_id)
    try:
        return await request.app.state.kc_status_service.list_library_files(
            domain_id=domain_id, collection_id=collection_id,
            query=q, status=status, page=page, page_size=page_size,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@catalog_router.get("/files/{document_version_id}/evidence")
async def list_file_evidence(
    domain_id: int, document_version_id: UUID, request: Request,
    collection_id: UUID, q: str | None = None,
    evidence_type: str | None = None, page_no: int | None = Query(None, ge=1),
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
):
    require_domain_match(request, domain_id)
    try:
        return await request.app.state.kc_status_service.list_file_evidence(
            domain_id=domain_id, collection_id=collection_id,
            document_version_id=document_version_id, query=q,
            evidence_type=evidence_type, page_no=page_no,
            page=page, page_size=page_size,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@catalog_router.get("/files/{document_version_id}/visual-assets")
async def list_file_visual_assets(
    domain_id: int, document_version_id: UUID, request: Request,
    collection_id: UUID, asset_type: str | None = None,
    page_no: int | None = Query(None, ge=1),
    page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
):
    require_domain_match(request, domain_id)
    try:
        return await request.app.state.kc_status_service.list_file_visual_assets(
            domain_id=domain_id, collection_id=collection_id,
            document_version_id=document_version_id, asset_type=asset_type,
            page_no=page_no, page=page, page_size=page_size,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@catalog_router.get(
    "/files/{document_version_id}/visual-assets/{visual_asset_id}/content"
)
async def get_file_visual_asset_content(
    domain_id: int, document_version_id: UUID, visual_asset_id: UUID,
    request: Request, collection_id: UUID,
):
    require_domain_match(request, domain_id)
    try:
        result = await (
            request.app.state.kc_status_service.get_file_visual_asset_content(
                domain_id=domain_id, collection_id=collection_id,
                document_version_id=document_version_id,
                visual_asset_id=visual_asset_id,
            )
        )
        stream = request.app.state.parser_artifact_store.stream(
            uri=result.payload_uri
        )
        return StreamingResponse(stream, media_type=result.mime_type)
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=410,
            detail={"code": "VISUAL_ASSET_GONE", "message": "视觉资产文件不存在"},
        ) from exc
