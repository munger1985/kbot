"""Scope-safe V2 ingestion and parsing status endpoints."""
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from knowledge_core.application.status import KnowledgeObjectNotFoundError


router = APIRouter(prefix="/api/v2/knowledge/domains/{domain_id}/bundles", tags=["Knowledge Core Status V2"])


@router.get("/{bundle_id}")
async def get_bundle_status(domain_id: int, bundle_id: int, request: Request):
    try:
        result = await request.app.state.kc_status_service.get_bundle(domain_id=domain_id, bundle_id=bundle_id)
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "BUNDLE_NOT_FOUND", "message": str(exc)}) from exc
    return asdict(result)


@router.get("/{bundle_id}/revisions/{bundle_revision_id}")
async def get_revision_status(domain_id: int, bundle_id: int, bundle_revision_id: int, request: Request):
    try:
        result = await request.app.state.kc_status_service.get_revision(
            domain_id=domain_id, bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "REVISION_NOT_FOUND", "message": str(exc)}) from exc
    return asdict(result)


@router.get("/{bundle_id}/revisions/{bundle_revision_id}/members")
async def get_revision_members(domain_id: int, bundle_id: int, bundle_revision_id: int, request: Request):
    try:
        result = await request.app.state.kc_status_service.get_revision(
            domain_id=domain_id, bundle_id=bundle_id, bundle_revision_id=bundle_revision_id,
            include_members=True,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": "REVISION_NOT_FOUND", "message": str(exc)}) from exc
    return {"bundle_revision_id": bundle_revision_id, "members": [asdict(item) for item in result.members or []]}
