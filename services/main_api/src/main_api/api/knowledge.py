"""知识检索 App 自有资料库的公开 BFF；不向浏览器暴露内部策略与存储句柄。"""

from typing import Annotated
from uuid import UUID
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from fastapi.routing import APIRoute
from platform_core.contracts.knowledge_app import (
    KnowledgeLibraryCreate, KnowledgeLibraryUpdate, KnowledgeReviewRequest,
    KnowledgeReprocessRequest, KnowledgeRetryIndexRequest, KnowledgeMaintenanceReceipt,
    KnowledgeIndexMaintenance, KnowledgeIndexRebuildRequest,
    KnowledgeSearchRequest, KnowledgeContinuationRequest, KnowledgeRelationsRequest,
    KnowledgeLibraryView, KnowledgeDocumentPage, KnowledgeUploadReceipt, KnowledgeActionReceipt,
    KnowledgeSearchResult, KnowledgeDirectoryQuery, KnowledgeMemberOption, KnowledgePassageContext,
    KnowledgePassageQuery, KnowledgeSearchHistoryPage, KnowledgeSearchHistoryDetail, parse_library_revision)
from platform_core.security import get_auth_context
from main_api.application import knowledge_library_members
from main_api.api.knowledge_parse_models import validate_parse_models
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


PERMISSIONS = {"create_library": "knowledge_core_manage", "update_library": "knowledge_core_manage", "delete_library": "knowledge_core_manage",
    "member_options": "knowledge_core_manage",
    "upload_document": "upload", "review_version": "review",
    "withdraw_version": "knowledge_core_manage", "cancel_version": "upload",
    "reprocess_version": "review", "retry_version_index": "knowledge_core_manage",
    "index_rebuild_status": "knowledge_core_manage", "operate_index_rebuild": "knowledge_core_manage"}


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
async def list_libraries(request: Request, management: bool = False):
    if management:
        await authorize_app_request(request, app_id="knowledge_retrieval", permission="knowledge_retrieval:knowledge_core_manage")
    return await client(request).libraries(management=management, auth_context=get_auth_context(request))


@router.post("/libraries", status_code=201, response_model=KnowledgeLibraryView)
async def create_library(payload: KnowledgeLibraryCreate, request: Request):
    await validate_parse_models(request, payload.parse_models)
    await validate_parse_models(request, payload.index_models)
    return await client(request).create_library(payload.model_dump(mode="json"),
                                                auth_context=get_auth_context(request))


@router.get("/libraries/{library_id}", response_model=KnowledgeLibraryView)
async def get_library(library_id: UUID, request: Request, management: bool = False):
    if management:
        await authorize_app_request(request, app_id="knowledge_retrieval", permission="knowledge_retrieval:knowledge_core_manage")
    return await client(request).library(library_id, management=management, auth_context=get_auth_context(request))


@router.delete("/libraries/{library_id}", response_model=KnowledgeActionReceipt)
async def delete_library(library_id: UUID, request: Request, if_match: Annotated[str | None, Header(alias="If-Match")] = None):
    if not if_match:
        raise HTTPException(428, {"code": "IF_MATCH_REQUIRED", "message": "请刷新知识库后重试"})
    try:
        version = parse_library_revision(if_match)
    except ValueError as exc:
        raise HTTPException(422, {"code": "IF_MATCH_INVALID", "message": str(exc)}) from exc
    return await client(request).delete_library(library_id, expected_revision=version, auth_context=get_auth_context(request))


@router.patch("/libraries/{library_id}", response_model=KnowledgeLibraryView)
async def update_library(library_id: UUID, payload: KnowledgeLibraryUpdate, request: Request):
    if payload.parse_models is not None or payload.index_models is not None:
        current = await client(request).library(library_id, management=True, auth_context=get_auth_context(request))
        await validate_parse_models(request, payload.parse_models, current=current.get("parse_models"))
        await validate_parse_models(request, payload.index_models, current=current.get("index_models"))
    if payload.grants is not None:
        await client(request).library(library_id, management=True, auth_context=get_auth_context(request))
        options = await knowledge_library_members.options(request.app.state, get_auth_context(request))
        if not knowledge_library_members.grants_allowed(payload.grants, options):
            raise HTTPException(409, {"code": "KNOWLEDGE_MEMBER_INELIGIBLE",
                "message": "请只保留当前工作域有效成员，并至少保留一名有管理资格的成员"})
    return await client(request).update_library(library_id, payload.model_dump(mode="json", exclude_unset=True),
                                               auth_context=get_auth_context(request))


@router.get("/libraries/{library_id}/index-rebuild", response_model=KnowledgeIndexMaintenance)
async def index_rebuild_status(library_id: UUID, request: Request, version_id: UUID | None = None):
    return await client(request).index_rebuild(library_id, auth_context=get_auth_context(request), version_id=version_id)


@router.post("/libraries/{library_id}/index-rebuild", response_model=KnowledgeIndexMaintenance)
async def operate_index_rebuild(library_id: UUID, payload: KnowledgeIndexRebuildRequest, request: Request):
    return await client(request).index_rebuild(library_id, auth_context=get_auth_context(request),
                                               payload=payload.model_dump(mode="json"))


@router.get("/libraries/{library_id}/member-options", response_model=list[KnowledgeMemberOption])
async def member_options(library_id: UUID, request: Request):
    await client(request).library(library_id, management=True, auth_context=get_auth_context(request))
    return await knowledge_library_members.options(request.app.state, get_auth_context(request))


@router.get("/libraries/{library_id}/documents", response_model=KnowledgeDocumentPage)
async def list_documents(library_id: UUID, request: Request, filters: Annotated[KnowledgeDirectoryQuery, Query()]):
    _, _, snapshot = await authorize_app_request(request, app_id="knowledge_retrieval",
        permission="knowledge_retrieval:use")
    governance = bool({"knowledge_retrieval:review", "knowledge_retrieval:knowledge_core_manage"} & set(snapshot.permissions))
    if filters.view == "reviews" and not governance:
        raise HTTPException(403, {"code": "KNOWLEDGE_REVIEW_DENIED", "message": "当前角色不能查看审核队列"})
    if filters.view == "processing" and not governance:
        await authorize_app_request(request, app_id="knowledge_retrieval", permission="knowledge_retrieval:upload")
        filters.mine = True
    return await client(request).files(library_id, **filters.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
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


@router.post("/libraries/{library_id}/versions/{version_id}:withdraw", response_model=KnowledgeActionReceipt)
async def withdraw_version(library_id: UUID, version_id: UUID, request: Request):
    return await client(request).version_action(library_id, version_id, "withdraw", auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:cancel", response_model=KnowledgeActionReceipt)
async def cancel_version(library_id: UUID, version_id: UUID, request: Request):
    return await client(request).version_action(library_id, version_id, "cancel", auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:reprocess", status_code=202,
             response_model=KnowledgeMaintenanceReceipt)
async def reprocess_version(library_id: UUID, version_id: UUID, payload: KnowledgeReprocessRequest,
                            request: Request):
    return await client(request).version_action(library_id, version_id, "reprocess",
        payload=payload.model_dump(mode="json"), auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/versions/{version_id}:retry-index", status_code=202,
             response_model=KnowledgeMaintenanceReceipt)
async def retry_version_index(library_id: UUID, version_id: UUID, payload: KnowledgeRetryIndexRequest,
                              request: Request):
    return await client(request).version_action(library_id, version_id, "retry-index",
        payload=payload.model_dump(mode="json"), auth_context=get_auth_context(request))


@router.post("/libraries/{library_id}/search", response_model=KnowledgeSearchResult)
async def search(library_id: UUID, payload: KnowledgeSearchRequest, request: Request):
    return await client(request).search(library_id, payload.model_dump(mode="json"), auth_context=get_auth_context(request))


@router.post("/retrievals/{retrieval_ref}:continue", response_model=KnowledgeSearchResult)
async def continue_results(retrieval_ref: UUID, payload: KnowledgeContinuationRequest, request: Request):
    return await client(request).continue_results(retrieval_ref, payload.model_dump(mode="json"),
                                                 auth_context=get_auth_context(request))


@router.get("/search-history", response_model=KnowledgeSearchHistoryPage)
async def search_history(request: Request, response: Response, limit: Annotated[int, Query(ge=1, le=50)] = 20,
                         offset: Annotated[int, Query(ge=0)] = 0):
    response.headers["Cache-Control"] = "no-store"
    return await client(request).history_page(auth_context=get_auth_context(request), limit=limit, offset=offset)


@router.get("/search-history/{identity}", response_model=KnowledgeSearchHistoryDetail)
async def search_history_detail(identity: UUID, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await client(request).history_detail(identity, auth_context=get_auth_context(request))


@router.delete("/search-history/{identity}", response_model=KnowledgeActionReceipt)
async def search_history_delete(identity: UUID, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await client(request).history_delete(identity, auth_context=get_auth_context(request))


@router.get("/search-history/{identity}/content/{key}")
async def search_history_content(identity: UUID, key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")], request: Request):
    download = await client(request).open_content(identity, key, auth_context=get_auth_context(request), history=True)
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))


@router.get("/search-history/{identity}/context/{key}", response_model=KnowledgePassageContext)
async def search_history_context(identity: UUID, key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
                                 query: Annotated[KnowledgePassageQuery, Query()], request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await client(request).passage_context(identity, key, **query.model_dump(),
                                                 auth_context=get_auth_context(request), history=True)


@router.get("/search-history/{identity}/context/{key}/pdf", response_class=StreamingResponse,
            responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}}})
async def search_history_pdf(identity: UUID, key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")], request: Request):
    download = await client(request).open_passage_pdf(identity, key, auth_context=get_auth_context(request), history=True)
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))


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
async def source_file(library_id: UUID, version_id: UUID, request: Request, preview: bool = False):
    download = await client(request).open_source(library_id, version_id, auth_context=get_auth_context(request), preview=preview)
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))


@router.get("/retrievals/{retrieval_ref}/context/{key}", response_model=KnowledgePassageContext)
async def passage_context(retrieval_ref: UUID, key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")],
                          query: Annotated[KnowledgePassageQuery, Query()], request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return await client(request).passage_context(retrieval_ref, key, **query.model_dump(),
                                                 auth_context=get_auth_context(request))


@router.get("/retrievals/{retrieval_ref}/context/{key}/pdf", response_class=StreamingResponse,
            responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}}})
async def passage_pdf(retrieval_ref: UUID, key: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")], request: Request):
    download = await client(request).open_passage_pdf(retrieval_ref, key, auth_context=get_auth_context(request))
    return StreamingResponse(download.blocks, media_type=download.mime_type, headers=download.headers,
                             background=BackgroundTask(download.close))
