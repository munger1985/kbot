"""正式知识资料库内部 API；业务入口只消费 App 独占存储。"""

from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from platform_core.contracts import AuthContext, ServiceIdentity
from platform_core.contracts.knowledge_app import (
    KnowledgeLibraryCreate, KnowledgeLibraryUpdate, KnowledgeReviewRequest,
    KnowledgeSearchRequest, KnowledgeContinuationRequest, KnowledgeRelationsRequest,
    KnowledgeLibraryView, KnowledgeDocumentPage, KnowledgeUploadReceipt, KnowledgeActionReceipt,
    KnowledgeSearchResult)
from ..domain.knowledge_retrieval import RetrievalAccess


class BoundedKnowledgeRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()
        async def limited(request):
            if request.method in {"POST", "PUT", "PATCH"}:
                maximum = 67108864 if request.url.path.endswith("/documents") else 12582912
                data = bytearray()
                async for block in request.stream():
                    if len(data) + len(block) > maximum:
                        raise HTTPException(413, {"code": "KNOWLEDGE_INPUT_TOO_LARGE"})
                    data.extend(block)
                request._body = bytes(data)
            return await handler(request)
        return limited


async def library_access(request: Request):
    identity = getattr(request.state, "service_identity", None)
    context = getattr(request.state, "auth_context", None)
    if (not isinstance(identity, ServiceIdentity) or not isinstance(context, AuthContext)
            or "knowledge_retrieval.manage" not in identity.scopes
            or "knowledge_retrieval.manage" not in context.scopes
            or context.app_id != "knowledge_retrieval" or context.domain_id is None):
        raise HTTPException(403, {"code": "KNOWLEDGE_LIBRARY_CONTEXT_REQUIRED"})
    try:
        return RetrievalAccess(int(context.domain_id), None, context.asserted_user_id or context.client_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(403, {"code": "KNOWLEDGE_LIBRARY_CONTEXT_INVALID"}) from exc


def create_library_router():
    router = APIRouter(prefix="/internal/v1/knowledge-retrieval/knowledge",
        tags=["app-knowledge-business"], route_class=BoundedKnowledgeRoute)

    def service(request):
        return request.app.state.knowledge_library_service

    @router.get("/libraries", response_model=list[KnowledgeLibraryView])
    async def listing(request: Request, access=Depends(library_access)):
        return await service(request).list(access)

    @router.post("/libraries", status_code=201, response_model=KnowledgeLibraryView)
    async def create(payload: KnowledgeLibraryCreate, request: Request, access=Depends(library_access)):
        return await service(request).create(access, **payload.model_dump())

    @router.get("/libraries/{library_id}", response_model=KnowledgeLibraryView)
    async def detail(library_id: UUID, request: Request, access=Depends(library_access)):
        return await service(request).get(access, library_id)

    @router.patch("/libraries/{library_id}", response_model=KnowledgeLibraryView)
    async def update(library_id: UUID, payload: KnowledgeLibraryUpdate, request: Request,
                     access=Depends(library_access)):
        return await service(request).update(access, library_id, expected_revision=payload.expected_revision,
            changes=payload.model_dump(exclude_unset=True, exclude={"expected_revision"}))

    @router.get("/libraries/{library_id}/documents", response_model=KnowledgeDocumentPage)
    async def files(library_id: UUID, request: Request, page: int = Query(1, ge=1),
                    page_size: int = Query(20, ge=1, le=100), access=Depends(library_access)):
        return await service(request).files(access, library_id, page=page, page_size=page_size)

    @router.post("/libraries/{library_id}/documents", status_code=202, response_model=KnowledgeUploadReceipt)
    async def upload(library_id: UUID, request: Request,
                     name: str = Query(min_length=1, max_length=512), access=Depends(library_access)):
        return await service(request).upload(access, library_id, name=name,
            mime_type=request.headers.get("Content-Type", "application/octet-stream").split(";")[0].lower(),
            raw=await request.body(), idempotency_key=request.headers.get("Idempotency-Key", ""))

    @router.post("/libraries/{library_id}/versions/{version_id}:review", response_model=KnowledgeActionReceipt)
    async def review(library_id: UUID, version_id: UUID, payload: KnowledgeReviewRequest,
                     request: Request, access=Depends(library_access)):
        return await service(request).review(access, library_id, version_id, **payload.model_dump())

    @router.post("/libraries/{library_id}/versions/{version_id}:publish", response_model=KnowledgeActionReceipt)
    async def publish(library_id: UUID, version_id: UUID, request: Request, access=Depends(library_access)):
        return await service(request).publish(access, library_id, version_id)

    @router.post("/libraries/{library_id}/versions/{version_id}:withdraw", response_model=KnowledgeActionReceipt)
    async def withdraw(library_id: UUID, version_id: UUID, request: Request, access=Depends(library_access)):
        return await service(request).withdraw(access, library_id, version_id)

    @router.post("/libraries/{library_id}/versions/{version_id}:cancel", response_model=KnowledgeActionReceipt)
    async def cancel(library_id: UUID, version_id: UUID, request: Request, access=Depends(library_access)):
        return await service(request).cancel(access, library_id, version_id)

    @router.get("/libraries/{library_id}/versions/{version_id}/source")
    async def source(library_id: UUID, version_id: UUID, request: Request, access=Depends(library_access)):
        record, stream = await service(request).source(access, library_id, version_id)
        return StreamingResponse(stream, media_type="application/octet-stream", headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Disposition": "attachment", "Content-Length": str(record["byte_count"])})

    @router.post("/libraries/{library_id}/search", response_model=KnowledgeSearchResult)
    async def search(library_id: UUID, payload: KnowledgeSearchRequest,
                     request: Request, access=Depends(library_access)):
        return await service(request).search(access, library_id, payload)

    @router.post("/results/{retrieval_ref}:continue", response_model=KnowledgeSearchResult)
    async def continuation(retrieval_ref: UUID, payload: KnowledgeContinuationRequest,
                           request: Request, access=Depends(library_access)):
        return await service(request).continue_search(access, str(retrieval_ref), payload.cursor)

    @router.post("/results/{retrieval_ref}/relations:continue", response_model=KnowledgeSearchResult)
    async def relations(retrieval_ref: UUID, payload: KnowledgeRelationsRequest,
                        request: Request, access=Depends(library_access)):
        return await service(request).continue_links(access, str(retrieval_ref), payload.anchor_key, payload.cursor)

    @router.get("/results/{retrieval_ref}/content/{stable_key}")
    async def content(retrieval_ref: UUID, stable_key: str, request: Request, access=Depends(library_access)):
        item, stream = await request.app.state.knowledge_retrieval_service.stream_content(
            access, retrieval_ref=str(retrieval_ref), stable_key=stable_key)
        return StreamingResponse(stream, media_type=item.mime_type, headers={"Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox"})

    return router
