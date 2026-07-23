"""Knowledge Core 能力的公开 BFF 契约。"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from platform_clients import KnowledgeCoreClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/knowledge",
    tags=["Knowledge"],
)


class CollectionCreateRequest(BaseModel):
    collection_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    display_name: str = Field(min_length=1, max_length=256)
    embedding_model_id: UUID
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionStatusRequest(BaseModel):
    status: str = Field(pattern=r"^(ACTIVE|DISABLED)$")


class CollectionBindingRequest(BaseModel):
    note: str | None = Field(default=None, max_length=1000)


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


@router.get("/collections/{collection_key}")
async def get_collection(collection_key: str, request: Request):
    context = get_auth_context(request)
    return await _client(request).get_collection(
        domain_id=_domain_id(request),
        collection_key=collection_key,
        auth_context=context,
    )


@router.patch("/collections/{collection_key}")
async def change_collection_status(
    collection_key: str,
    payload: CollectionStatusRequest,
    request: Request,
):
    context = get_auth_context(request)
    return await _client(request).change_collection_status(
        domain_id=_domain_id(request),
        collection_key=collection_key,
        status=payload.status,
        auth_context=context,
    )


@router.delete(
    "/collections/{collection_key}",
    status_code=status.HTTP_202_ACCEPTED,
)
async def delete_collection(collection_key: str, request: Request):
    context = get_auth_context(request)
    return await _client(request).delete_collection(
        domain_id=_domain_id(request),
        collection_key=collection_key,
        auth_context=context,
    )


@router.put(
    "/agents/{agent_id}/collections/{collection_key}/binding",
)
async def bind_collection(
    agent_id: UUID,
    collection_key: str,
    request: Request,
    payload: CollectionBindingRequest | None = None,
):
    context = get_auth_context(request)
    return await _client(request).bind_collection(
        domain_id=_domain_id(request),
        agent_id=agent_id,
        collection_key=collection_key,
        note=payload.note if payload is not None else None,
        auth_context=context,
    )


@router.delete(
    "/agents/{agent_id}/collections/{collection_key}/binding",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def unbind_collection(
    agent_id: UUID,
    collection_key: str,
    request: Request,
) -> Response:
    context = get_auth_context(request)
    await _client(request).unbind_collection(
        domain_id=_domain_id(request),
        agent_id=agent_id,
        collection_key=collection_key,
        auth_context=context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/agents/{agent_id}/collection-bindings")
async def list_agent_bindings(agent_id: UUID, request: Request):
    context = get_auth_context(request)
    return await _client(request).list_agent_bindings(
        domain_id=_domain_id(request),
        agent_id=agent_id,
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


async def _ingest(
    *,
    request: Request,
    collection_key: str,
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
        collection_key=collection_key,
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
                f"{PUBLIC_API_V1}/knowledge/bundles/"
                f"{payload['bundle_id']}"
            ),
        }
    return JSONResponse(
        status_code=upstream.status_code,
        content=payload,
    )


@router.post(
    "/collections/{collection_key}/ingestions/km-assets",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": {"type": "object"}}},
        }
    },
)
async def ingest_km_asset(collection_key: str, request: Request):
    return await _ingest(
        request=request,
        collection_key=collection_key,
        intake_kind="km-assets",
    )


@router.post(
    "/collections/{collection_key}/ingestions/user-files",
    status_code=status.HTTP_202_ACCEPTED,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"multipart/form-data": {"schema": {"type": "object"}}},
        }
    },
)
async def ingest_user_files(collection_key: str, request: Request):
    return await _ingest(
        request=request,
        collection_key=collection_key,
        intake_kind="user-files",
    )
