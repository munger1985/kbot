"""Knowledge Core 能力的公开 BFF 契约。"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from platform_clients import KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context
from main_api.application import AccessControlService, AccessDeniedError


async def _require_knowledge_access(request: Request) -> None:
    context = get_auth_context(request)
    actor_id = context.asserted_user_id or context.client_id
    relative = request.url.path.removeprefix(
        f"{PUBLIC_API_V1}/apps/knowledge-retrieval/knowledge"
    )
    permission = "knowledge_retrieval:use"
    if "/approval" in relative:
        permission = "knowledge_retrieval:review"
    elif relative.startswith("/agents/"):
        permission = "knowledge_retrieval:agent_manage"
    elif "/ingestions/user-files" in relative:
        permission = "knowledge_retrieval:upload"
    elif request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        permission = (
            "knowledge_retrieval:knowledge_manage"
        )
    try:
        await request.app.state.access_control_service.require(
            app_id="knowledge_retrieval",
            domain_id=int(context.domain_id or "0"),
            user_id=actor_id,
            permission_code=permission,
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403, {"code": "APP_PERMISSION_DENIED", "permission": permission}
        ) from exc


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/knowledge-retrieval/knowledge",
    tags=["Knowledge"],
    dependencies=[Depends(_require_knowledge_access)],
)


class CollectionCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=256)
    models: dict[str, UUID]
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(ACTIVE|DISABLED)$")


class CollectionModelsRequest(BaseModel):
    models: dict[str, UUID]
    expected_row_version: int = Field(ge=1)


class IntakeReviewRequest(BaseModel):
    decision: str = Field(pattern=r"^(APPROVE|REJECT)$")
    comment: str | None = Field(default=None, max_length=1000)


class RevisionReprocessRequest(BaseModel):
    collection_id: UUID
    document_version_id: UUID | None = None


def _domain_id(request: Request) -> int:
    context = get_auth_context(request)
    if context.domain_id is None:
        raise RuntimeError("已认证请求缺少 Domain")
    return int(context.domain_id)


def _client(request: Request) -> KnowledgeCoreClient:
    client = getattr(request.app.state, "knowledge_core_client", None)
    if client is None:
        raise RuntimeError("Knowledge Core Client 尚未初始化")
    return cast(KnowledgeCoreClient, client)


@router.get("/collections")
async def list_collections(request: Request):
    context = get_auth_context(request)
    return await _client(request).list_collections(
        domain_id=_domain_id(request),
        auth_context=context,
    )


@router.post("/collections", status_code=status.HTTP_201_CREATED)
async def create_collection(
    payload: CollectionCreateRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).create_collection(
        domain_id=_domain_id(request),
        payload=payload.model_dump(mode="json"),
        auth_context=context,
    )


@router.get("/collections/{collection_id}")
async def get_collection(collection_id: UUID, request: Request):
    context = get_auth_context(request)
    return await _client(request).get_collection(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        auth_context=context,
    )


@router.patch("/collections/{collection_id}")
async def change_collection_status(
    collection_id: UUID,
    payload: CollectionStatusRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).change_collection_status(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        status=payload.status,
        auth_context=context,
    )


@router.put("/collections/{collection_id}/models")
async def update_collection_models(
    collection_id: UUID,
    payload: CollectionModelsRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).update_collection_models(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        payload=payload.model_dump(mode="json"),
        auth_context=context,
    )


@router.delete(
    "/collections/{collection_id}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_collection(collection_id: UUID, request: Request):
    context = get_auth_context(request)
    return await _client(request).delete_collection(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        auth_context=context,
    )


@router.get("/bundles/{bundle_id}")
async def get_bundle_status(bundle_id: UUID, request: Request):
    context = get_auth_context(request)
    return await _client(request).get_bundle_status(
        domain_id=_domain_id(request),
        bundle_id=bundle_id,
        auth_context=context,
    )


@router.get("/bundles/{bundle_id}/revisions/{bundle_revision_id}")
async def get_revision_status(
    bundle_id: UUID,
    bundle_revision_id: UUID,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).get_revision_status(
        domain_id=_domain_id(request),
        bundle_id=bundle_id,
        bundle_revision_id=bundle_revision_id,
        include_members=False,
        auth_context=context,
    )


@router.get(
    "/bundles/{bundle_id}/revisions/{bundle_revision_id}/members",
)
async def get_revision_members(
    bundle_id: UUID,
    bundle_revision_id: UUID,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).get_revision_status(
        domain_id=_domain_id(request),
        bundle_id=bundle_id,
        bundle_revision_id=bundle_revision_id,
        include_members=True,
        auth_context=context,
    )


@router.post(
    "/bundles/{bundle_id}/revisions/{bundle_revision_id}/reprocess",
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_revision(
    bundle_id: UUID,
    bundle_revision_id: UUID,
    payload: RevisionReprocessRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).reprocess_revision(
        domain_id=_domain_id(request),
        collection_id=payload.collection_id,
        bundle_id=bundle_id,
        bundle_revision_id=bundle_revision_id,
        document_version_id=payload.document_version_id,
        auth_context=context,
    )


@router.get("/collections/{collection_id}/approvals")
async def list_pending_approvals(
    collection_id: UUID,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).list_pending_approvals(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        auth_context=context,
    )


@router.post(
    "/collections/{collection_id}/bundle-revisions/"
    "{bundle_revision_id}/approval",
)
async def review_user_intake(
    collection_id: UUID,
    bundle_revision_id: UUID,
    payload: IntakeReviewRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).review_user_intake(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        bundle_revision_id=bundle_revision_id,
        decision=payload.decision,
        comment=payload.comment,
        auth_context=context,
    )


async def _ingest(
    *,
    request: Request,
    collection_id: UUID,
    intake_kind: str,
) -> JSONResponse:
    content_type = request.headers.get("Content-Type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(
            status_code=415,
            detail={
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": "入库接口只接受 multipart/form-data",
            },
        )
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if not idempotency_key:
        raise HTTPException(
            status_code=428,
            detail={
                "code": "IDEMPOTENCY_KEY_REQUIRED",
                "message": "缺少 Idempotency-Key Header",
            },
        )
    context = get_auth_context(request)
    upstream = await _client(request).ingest_multipart(
        domain_id=_domain_id(request),
        collection_id=collection_id,
        intake_kind=intake_kind,
        content_type=content_type,
        body=request.stream(),
        idempotency_key=idempotency_key,
        auth_context=context,
    )
    payload = upstream.payload
    if isinstance(payload, dict) and "bundle_id" in payload:
        payload = {
            **payload,
            "status_url": (
                f"{PUBLIC_API_V1}/apps/knowledge-retrieval/knowledge/bundles/"
                f"{payload['bundle_id']}"
            ),
        }
    return JSONResponse(
        status_code=upstream.status_code,
        content=payload,
    )


@router.post(
    "/collections/{collection_id}/ingestions/km-assets",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": {"type": "object"}}},
        }
    },
)
async def ingest_km_asset(collection_id: UUID, request: Request):
    return await _ingest(
        request=request,
        collection_id=collection_id,
        intake_kind="km-assets",
    )


@router.post(
    "/collections/{collection_id}/ingestions/user-files",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": {"type": "object"}}},
        }
    },
)
async def ingest_user_files(collection_id: UUID, request: Request):
    return await _ingest(
        request=request,
        collection_id=collection_id,
        intake_kind="user-files",
    )
