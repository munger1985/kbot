"""知识检索 App 自有资料库的公开 BFF；不向浏览器暴露内部策略与存储句柄。"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.routing import APIRoute
from platform_core.contracts.knowledge_app import (
    KnowledgeLibraryCreate, KnowledgeLibraryUpdate, KnowledgeReviewRequest,
    KnowledgeSearchRequest, KnowledgeContinuationRequest, KnowledgeRelationsRequest,
    KnowledgeLibraryView, KnowledgeDocumentPage, KnowledgeUploadReceipt, KnowledgeActionReceipt,
    KnowledgeSearchResult)
from platform_core.security import get_auth_context
from main_api.application import authorize_app_request


class KnowledgeBodyLimitRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def limited(request):
            if request.method in {"POST", "PUT", "PATCH"}:
                maximum = 67108864 if request.url.path.endswith("/documents") else 12582912
                raw = bytearray()
                async for block in request.stream():
                    if len(raw) + len(block) > maximum:
                        raise HTTPException(413, {"code": "KNOWLEDGE_INPUT_TOO_LARGE"})
                    raw.extend(block)
                request._body = bytes(raw)
            return await handler(request)
        return limited


PERMISSIONS = {"create_library": "knowledge_core_manage", "update_library": "knowledge_core_manage",
    "upload_document": "upload", "review_version": "review", "publish_version": "knowledge_core_manage",
    "withdraw_version": "knowledge_core_manage", "cancel_version": "upload"}


async def require_access(request: Request):
    name = request.scope["endpoint"].__name__
    permission = PERMISSIONS.get(name, "use")
    await authorize_app_request(request, app_id="knowledge_retrieval",
                                permission="knowledge_retrieval:" + permission)


router = APIRouter(prefix="/api/v1/apps/knowledge-retrieval/knowledge", tags=["Knowledge Libraries"],
    route_class=KnowledgeBodyLimitRoute, dependencies=[Depends(require_access)])


def client(request):
    return request.app.state.knowledge_library_client


@router.get("/libraries", response_model=list[KnowledgeLibraryView])
async def list_libraries(request: Request):
    return await client(request).libraries(auth_context=get_auth_context(request))


@router.post("/libraries", status_code=201, response_model=KnowledgeLibraryView)
async def create_library(payload: KnowledgeLibraryCreate, request: Request):
    return await client(request).create_library(payload.model_dump(mode="json"),
                                                auth_context=get_auth_context(request))


@router.get("/libraries/{library_id}", response_model=KnowledgeLibraryView)
async def get_library(library_id: UUID, request: Request):
    return await client(request).library(library_id, auth_context=get_auth_context(request))


@router.patch("/libraries/{library_id}", response_model=KnowledgeLibraryView)
async def update_library(library_id: UUID, payload: KnowledgeLibraryUpdate, request: Request):
    return await client(request).update_library(library_id, payload.model_dump(mode="json", exclude_unset=True),
                                               auth_context=get_auth_context(request))


@router.get("/libraries/{library_id}/documents", response_model=KnowledgeDocumentPage)
async def list_documents(library_id: UUID, request: Request, page: int = Query(1, ge=1),
                         page_size: int = Query(20, ge=1, le=100)):
    return await client(request).files(library_id, page=page, page_size=page_size,
                                      auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/documents", status_code=202, response_model=KnowledgeUploadReceipt)
async def upload_document(library_id: UUID, request: Request, name: str = Query(min_length=1, max_length=512)):
    key = request.headers.get("Idempotency-Key", "").strip()
    if not key:
        raise HTTPException(428, {"code": "IDEMPOTENCY_KEY_REQUIRED"})
    return await client(request).upload(library_id, name=name,
        mime_type=request.headers.get("Content-Type", "application/octet-stream").split(";")[0],
        body=request.stream(), idempotency_key=key, auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:review", response_model=KnowledgeActionReceipt)
async def review_version(library_id: UUID, version_id: UUID, payload: KnowledgeReviewRequest, request: Request):
    return await client(request).version_action(library_id, version_id, "review",
        payload=payload.model_dump(mode="json"), auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:publish", response_model=KnowledgeActionReceipt)
async def publish_version(library_id: UUID, version_id: UUID, request: Request):
    return await client(request).version_action(library_id, version_id, "publish", auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:withdraw", response_model=KnowledgeActionReceipt)
async def withdraw_version(library_id: UUID, version_id: UUID, request: Request):
    return await client(request).version_action(library_id, version_id, "withdraw", auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:cancel", response_model=KnowledgeActionReceipt)
async def cancel_version(library_id: UUID, version_id: UUID, request: Request):
    return await client(request).version_action(library_id, version_id, "cancel", auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/search", response_model=KnowledgeSearchResult)
async def search(library_id: UUID, payload: KnowledgeSearchRequest, request: Request):
    return await client(request).search(library_id, payload.model_dump(mode="json"), auth_context=get_auth_context(request))


@router.post("/retrievals/{retrieval_ref}:continue", response_model=KnowledgeSearchResult)
async def continue_results(retrieval_ref: UUID, payload: KnowledgeContinuationRequest, request: Request):
    return await client(request).continue_results(retrieval_ref, payload.model_dump(mode="json"),
                                                 auth_context=get_auth_context(request))


@router.post("/retrievals/{retrieval_ref}/relations:continue", response_model=KnowledgeSearchResult)
async def continue_relations(retrieval_ref: UUID, payload: KnowledgeRelationsRequest, request: Request):
    return await client(request).continue_results(retrieval_ref, payload.model_dump(mode="json"),
                                                 relations=True, auth_context=get_auth_context(request))


@router.get("/retrievals/{retrieval_ref}/content/{key}")
async def evidence_content(retrieval_ref: UUID, key: str, request: Request):
    download = await client(request).open_content(retrieval_ref, key, auth_context=get_auth_context(request))
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))


@router.get("/libraries/{library_id}/versions/{version_id}/source")
async def source_file(library_id: UUID, version_id: UUID, request: Request):
    download = await client(request).open_source(library_id, version_id, auth_context=get_auth_context(request))
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))
