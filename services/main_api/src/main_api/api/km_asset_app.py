"""KM Asset App 公开 BFF API。"""

from typing import Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    KM_PORTAL_DOMAIN_NAME,
    UserAuthService,
)
from platform_clients import AgentRuntimeClient, KmAssetClient, KnowledgeCoreClient
from platform_core.contracts import ConversationQueryImage, PUBLIC_API_V1, UpdateConversationRequest
from fastapi import Response
from main_api.api.runs import (
    DocumentReferencePreview,
    _DocumentReference,
    _document_locator,
    _effective_security_level,
    _event_stream,
    _parse_cursor,
    _preview_type,
    _reference_not_found,
)
from main_api.api.models import ModelCatalogItem, load_model_catalog
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/apps/km-asset", tags=["KM Asset App"])
KM_ASSET_COLLECTION_NAME = "assets"


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KmLoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class KmPasswordChangePayload(_Payload):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        if not (
            any(char.islower() for char in value)
            and any(char.isupper() for char in value)
            and any(char.isdigit() for char in value)
            and any(not char.isalnum() for char in value)
        ):
            raise ValueError("新密码必须同时包含大小写字母、数字和特殊字符")
        return value


class SourceCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    metadb_endpoint: AnyHttpUrl
    metadb_credentials: dict[str, str] = Field(min_length=2, max_length=2)
    sharepoint_credentials: dict[str, str] = Field(min_length=3, max_length=3)
    sharepoint_site_path: str = Field(min_length=1, max_length=512)
    poll_interval_seconds: int = Field(default=60, ge=10, le=86400)
    batch_size: int = Field(default=100, ge=1, le=1000)

    @field_validator("metadb_credentials")
    @classmethod
    def validate_metadb_credentials(cls, value):
        if set(value) != {"username", "password"} or any(
            not str(item).strip() for item in value.values()
        ):
            raise ValueError("MetaDB 凭据必须包含 username 和 password")
        return value

    @field_validator("sharepoint_credentials")
    @classmethod
    def validate_sharepoint_credentials(cls, value):
        if set(value) != {"tenant_id", "client_id", "client_secret"} or any(
            not str(item).strip() for item in value.values()
        ):
            raise ValueError("SharePoint 凭据字段不完整")
        return value


class SourceUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    metadb_endpoint: AnyHttpUrl | None = None
    metadb_credentials: dict[str, str] | None = Field(default=None, min_length=2, max_length=2)
    sharepoint_credentials: dict[str, str] | None = Field(default=None, min_length=3, max_length=3)
    sharepoint_site_path: str | None = Field(default=None, min_length=1, max_length=512)
    auto_sync_enabled: bool | None = None
    poll_interval_seconds: int | None = Field(default=None, ge=10, le=86400)
    batch_size: int | None = Field(default=None, ge=1, le=1000)

    @field_validator("metadb_credentials")
    @classmethod
    def validate_metadb_credentials(cls, value):
        if value is not None and (
            set(value) != {"username", "password"}
            or any(not str(item).strip() for item in value.values())
        ):
            raise ValueError("MetaDB 凭据必须包含 username 和 password")
        return value

    @field_validator("sharepoint_credentials")
    @classmethod
    def validate_sharepoint_credentials(cls, value):
        if value is not None and (
            set(value) != {"tenant_id", "client_id", "client_secret"}
            or any(not str(item).strip() for item in value.values())
        ):
            raise ValueError("SharePoint 凭据字段不完整")
        return value

    @model_validator(mode="after")
    def require_change(self):
        if not self.model_dump(exclude={"expected_row_version"}, exclude_none=True):
            raise ValueError("至少提供一个需要修改的来源字段")
        return self


class VersionPayload(_Payload):
    expected_row_version: int = Field(ge=1)


class AgentCreatePayload(_Payload):
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID] = Field(default_factory=dict)
    do_rerank: bool = False
    instruction: str | None = Field(default=None, max_length=32000)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AgentActivatePayload(_Payload):
    expected_row_version: int = Field(ge=1)


class AgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID] = Field(default_factory=dict)
    do_rerank: bool = False
    instruction: str | None = Field(default=None, max_length=32000)


class ConversationCreatePayload(_Payload):
    agent_id: UUID
    title: str | None = Field(default=None, min_length=1, max_length=512)
    retention_policy: str = Field(default="DEFAULT", pattern=r"^(DEFAULT|KEEP_FOREVER|DAYS_30|DAYS_90|DAYS_365)$")


class ConversationTurnPayload(_Payload):
    input: str = Field(min_length=1, max_length=32000)
    expected_conversation_version: int = Field(ge=1)
    collection_ids: tuple[UUID, ...] = ()
    client_metadata: dict[str, object] = Field(default_factory=dict)
    images: tuple[ConversationQueryImage, ...] = Field(
        default=(), max_length=8
    )


def _domain_actor(request: Request) -> tuple[int, str]:
    context = get_auth_context(request)
    if context.app_id and context.app_id != "km_asset":
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH"})
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"}) from exc
    if domain_id < 1:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, context.asserted_user_id or context.client_id


async def _require(request: Request, permission: str) -> int:
    domain_id, actor_id = _domain_actor(request)
    service = cast(AccessControlService, request.app.state.access_control_service)
    try:
        await service.require(app_id="km_asset", domain_id=domain_id, user_id=actor_id, permission_code=permission)
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": permission}) from exc
    return domain_id


def _client(request: Request) -> KmAssetClient:
    return cast(KmAssetClient, request.app.state.km_asset_client)


def _km_turn_receipt(receipt: dict) -> dict:
    """将运行时收据投影为 KM App 自有的公开访问地址。"""
    projected = dict(receipt)
    run_id = projected.get("run_id")
    if run_id:
        projected["events_url"] = (
            f"{PUBLIC_API_V1}/apps/km-asset/runs/{run_id}/events"
        )
    return projected


async def _fixed_collection_id(request: Request, *, domain_id: int) -> UUID:
    """解析 KM 固定 Collection，拒绝缺失、停用或重复配置。"""
    catalog = await cast(
        KnowledgeCoreClient, request.app.state.knowledge_core_client
    ).list_collections(
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    matches = [
        item
        for item in catalog.get("collections", [])
        if item.get("display_name") == KM_ASSET_COLLECTION_NAME
        and item.get("status") == "ACTIVE"
    ]
    if len(matches) != 1:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "KM_FIXED_COLLECTION_UNAVAILABLE",
                "message": "KM 固定 Collection assets 尚未初始化、未启用或存在重复记录",
            },
        )
    return UUID(str(matches[0]["collection_id"]))


async def _km_conversation(request: Request, conversation_id: UUID, domain_id: int):
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    conversation = await runtime.get_conversation(
        conversation_id=conversation_id,
        auth_context=request.state.auth_context,
    )
    await _client(request).get_agent(
        agent_id=UUID(str(conversation["agent_id"])),
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    return conversation


async def _km_run(request: Request, run_id: UUID, domain_id: int):
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    run = await runtime.get_run(run_id=run_id, auth_context=request.state.auth_context)
    await _client(request).get_agent(
        agent_id=UUID(str(run["agent_id"])),
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    return run


@router.post("/auth/login")
async def login(payload: KmLoginPayload, request: Request):
    """使用平台用户凭据进入固定的 KM Portal Domain。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.login_for_domain_name(
        user_id=payload.user_id.strip(),
        password=payload.password,
        domain_name=KM_PORTAL_DOMAIN_NAME,
    )


@router.post("/auth/password")
async def change_password(payload: KmPasswordChangePayload, request: Request):
    """首次登录或后续主动修改 KM 本地用户密码。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    claims = request.state.user_token_claims
    return await service.change_password(
        claims=claims,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


@router.get("/access")
async def get_access(request: Request):
    domain_id, actor_id = _domain_actor(request)
    snapshot = await cast(AccessControlService, request.app.state.access_control_service).snapshot(app_id="km_asset", domain_id=domain_id, user_id=actor_id)
    if "km_asset:use" not in snapshot.permissions:
        raise HTTPException(
            403,
            {
                "code": "APP_PERMISSION_DENIED",
                "permission": "km_asset:use",
            },
        )
    return {"app_id": snapshot.app_id, "domain_id": snapshot.domain_id, "user_id": snapshot.user_id, "roles": snapshot.roles, "permissions": sorted(snapshot.permissions)}


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_km_model_catalog(request: Request):
    """在 KM Token 的访问边界内返回创建 KM Agent 所需模型。"""
    await _require(request, "km_asset:agent_manage")
    return await load_model_catalog(request)


@router.get("/sources")
async def list_sources(request: Request):
    domain_id = await _require(request, "km_asset:source_manage")
    return await _client(request).list_sources(domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/sources", status_code=status.HTTP_201_CREATED)
async def create_source(payload: SourceCreatePayload, request: Request):
    domain_id = await _require(request, "km_asset:source_manage")
    collection_id = await _fixed_collection_id(request, domain_id=domain_id)
    return await _client(request).create_source(
        payload={
            "domain_id": domain_id,
            "collection_id": str(collection_id),
            **payload.model_dump(mode="json"),
        },
        auth_context=request.state.auth_context,
    )


@router.patch("/sources/{source_id}")
async def update_source(source_id: UUID, payload: SourceUpdatePayload, request: Request):
    domain_id = await _require(request, "km_asset:source_manage")
    return await _client(request).update_source(
        source_id=source_id,
        payload={"domain_id": domain_id, **payload.model_dump(mode="json", exclude_none=True)},
        auth_context=request.state.auth_context,
    )


@router.post("/sources/{source_id}/activate")
async def activate_source(source_id: UUID, payload: VersionPayload, request: Request):
    domain_id = await _require(request, "km_asset:source_manage")
    return await _client(request).activate_source(source_id=source_id, payload={"domain_id": domain_id, **payload.model_dump()}, auth_context=request.state.auth_context)


@router.post("/sources/{source_id}/sync", status_code=status.HTTP_202_ACCEPTED)
async def sync_source(source_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).sync_source(source_id=source_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.get("/sources/{source_id}/metadb/assets")
async def list_metadb_assets(source_id: UUID, request: Request, processed: Literal["N", "Y", "F"] = "N", offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
    domain_id = await _require(request, "km_asset:source_manage")
    return await _client(request).list_metadb_assets(source_id=source_id, domain_id=domain_id, processed=processed, offset=offset, limit=limit, auth_context=request.state.auth_context)


@router.post("/sources/{source_id}/metadb/assets/{external_asset_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_metadb_asset(source_id: UUID, external_asset_id: str, request: Request):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).retry_metadb_asset(
        source_id=source_id,
        external_asset_id=external_asset_id,
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )


@router.get("/sources/{source_id}/data-model")
async def get_data_model(source_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:data_manage")
    return await _client(request).data_model(source_id=source_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/sources/{source_id}/data-model/reconcile")
async def reconcile_data_model(source_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:data_manage")
    return await _client(request).reconcile_data_model(source_id=source_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.get("/assets")
async def list_assets(request: Request, source_id: UUID | None = None, ingestion_status: str | None = None, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
    domain_id = await _require(request, "km_asset:use")
    return await _client(request).list_assets(domain_id=domain_id, source_id=source_id, ingestion_status=ingestion_status, offset=offset, limit=limit, auth_context=request.state.auth_context)


@router.get("/assets/{km_asset_id}")
async def get_asset(km_asset_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    return await _client(request).get_asset(km_asset_id=km_asset_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/assets/{km_asset_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_asset(km_asset_id: UUID, payload: VersionPayload, request: Request):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).retry_asset(km_asset_id=km_asset_id, payload={"domain_id": domain_id, **payload.model_dump()}, auth_context=request.state.auth_context)


@router.post("/assets/{km_asset_id}/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_asset(km_asset_id: UUID, payload: VersionPayload, request: Request):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).reindex_asset(
        km_asset_id=km_asset_id,
        payload={"domain_id": domain_id, **payload.model_dump()},
        auth_context=request.state.auth_context,
    )


@router.get("/jobs")
async def list_jobs(request: Request, source_id: UUID | None = None, limit: int = Query(default=100, ge=1, le=500)):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).list_jobs(domain_id=domain_id, source_id=source_id, limit=limit, auth_context=request.state.auth_context)


@router.get("/agents")
async def list_agents(request: Request):
    domain_id = await _require(request, "km_asset:use")
    return await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreatePayload, request: Request):
    domain_id = await _require(request, "km_asset:agent_manage")
    return await _client(request).create_agent(payload={"domain_id": domain_id, **payload.model_dump(mode="json")}, auth_context=request.state.auth_context)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    return await _client(request).get_agent(agent_id=agent_id, domain_id=domain_id, auth_context=request.state.auth_context)


@router.patch("/agents/{agent_id}")
async def update_agent(
    agent_id: UUID, payload: AgentUpdatePayload, request: Request
):
    domain_id = await _require(request, "km_asset:agent_manage")
    return await _client(request).update_agent(
        agent_id=agent_id,
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
    )


@router.post("/agents/{agent_id}/activate")
async def activate_agent(agent_id: UUID, payload: AgentActivatePayload, request: Request):
    domain_id = await _require(request, "km_asset:agent_manage")
    return await _client(request).activate_agent(
        agent_id=agent_id,
        payload={"domain_id": domain_id, **payload.model_dump()},
        auth_context=request.state.auth_context,
    )


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_conversation(payload: ConversationCreatePayload, request: Request):
    domain_id = await _require(request, "km_asset:use")
    spec = await _client(request).execution_spec(agent_id=payload.agent_id, domain_id=domain_id, auth_context=request.state.auth_context)
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    return await runtime.create_conversation(payload={**payload.model_dump(mode="json"), "execution_spec": spec}, auth_context=request.state.auth_context)


@router.get("/conversations")
async def list_conversations(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    domain_id = await _require(request, "km_asset:use")
    agents = await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)
    agent_ids = {str(item["agent_id"]) for item in agents}
    rows = await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).list_conversations(limit=200, auth_context=request.state.auth_context)
    return [item for item in rows if str(item.get("agent_id")) in agent_ids][:limit]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    return await _km_conversation(request, conversation_id, domain_id)


@router.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: UUID, payload: UpdateConversationRequest, request: Request):
    domain_id = await _require(request, "km_asset:use")
    await _km_conversation(request, conversation_id, domain_id)
    return await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).update_conversation(conversation_id=conversation_id, payload=payload.model_dump(mode="json"), auth_context=request.state.auth_context)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, request: Request, expected_row_version: int = Query(ge=1)):
    domain_id = await _require(request, "km_asset:use")
    await _km_conversation(request, conversation_id, domain_id)
    await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).delete_conversation(conversation_id=conversation_id, expected_row_version=expected_row_version, auth_context=request.state.auth_context)
    return Response(status_code=204)


@router.get("/runs/{run_id}")
async def get_run(run_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    return await _km_run(request, run_id, domain_id)


@router.get("/runs/{run_id}/result")
async def get_run_result(run_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    await _km_run(request, run_id, domain_id)
    return await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).get_result(run_id=run_id, auth_context=request.state.auth_context)


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    domain_id = await _require(request, "km_asset:use")
    cursor = _parse_cursor(last_event_id)
    summary = await _km_run(request, run_id, domain_id)
    if cursor > int(summary["event_cursor"]):
        raise HTTPException(400, {"code": "AGENT_EVENT_CURSOR_INVALID", "message": "Last-Event-ID 超过当前 Run 事件游标"})
    return StreamingResponse(
        _event_stream(run_id=run_id, request=request, cursor=cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _document_reference(request: Request, run_id: UUID, citation_label: str) -> _DocumentReference:
    await _km_run(request, run_id, int(request.state.auth_context.domain_id))
    artifact = await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).get_result(run_id=run_id, auth_context=request.state.auth_context)
    payload = artifact.get("payload")
    references = payload.get("references") if isinstance(payload, dict) else None
    raw = next((item for item in references or [] if isinstance(item, dict) and item.get("reference_type") == "DOCUMENT" and item.get("citation_label") == citation_label), None)
    if raw is None:
        raise _reference_not_found()
    try:
        return _DocumentReference.model_validate(raw)
    except ValueError as exc:
        raise HTTPException(409, {"code": "DOCUMENT_REFERENCE_INVALID", "message": "Run 引用缺少不可变文档定位信息"}) from exc


@router.get("/runs/{run_id}/references/{citation_label}/preview", response_model=DocumentReferencePreview)
async def get_reference_preview(run_id: UUID, citation_label: str, request: Request):
    await _require(request, "km_asset:use")
    reference = await _document_reference(request, run_id, citation_label)
    preview = await cast(KnowledgeCoreClient, request.app.state.knowledge_core_client).get_bundle_revision_preview(domain_id=int(request.state.auth_context.domain_id), collection_id=reference.collection_id, bundle_id=reference.bundle_id, bundle_revision_id=reference.bundle_revision_id, auth_context=request.state.auth_context)
    source_file = next((item for item in preview.get("files", []) if str(item.get("document_version_id")) == str(reference.document_version_id) and bool(item.get("preview_available"))), None)
    if source_file is None:
        raise _reference_not_found()
    mime_type = str(source_file.get("detected_mime_type") or source_file.get("declared_mime_type") or "application/octet-stream").split(";", 1)[0].strip().lower()
    page_no, page_end, bbox = _document_locator(reference)
    return DocumentReferencePreview(citation_label=reference.citation_label, title=reference.title, mime_type=mime_type, preview_type=_preview_type(mime_type), page_no=page_no, page_end=page_end, bbox=bbox, content_url=f"{PUBLIC_API_V1}/apps/km-asset/runs/{run_id}/references/{reference.citation_label}/content")


@router.get("/runs/{run_id}/references/{citation_label}/content")
async def stream_reference_content(run_id: UUID, citation_label: str, request: Request, range_header: str | None = Header(default=None, alias="Range")):
    await _require(request, "km_asset:use")
    reference = await _document_reference(request, run_id, citation_label)
    upstream = await cast(KnowledgeCoreClient, request.app.state.knowledge_core_client).stream_source_file(domain_id=int(request.state.auth_context.domain_id), collection_id=reference.collection_id, bundle_id=reference.bundle_id, bundle_revision_id=reference.bundle_revision_id, document_version_id=reference.document_version_id, range_header=range_header, auth_context=request.state.auth_context)
    forwarded = {name: upstream.headers[name] for name in ("accept-ranges", "cache-control", "content-disposition", "content-length", "content-range", "content-security-policy", "x-content-type-options") if name in upstream.headers}
    return StreamingResponse(upstream.body, status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/octet-stream"), headers=forwarded)


@router.post("/conversations/{conversation_id}/turns", status_code=status.HTTP_202_ACCEPTED)
async def create_conversation_turn(conversation_id: UUID, payload: ConversationTurnPayload, request: Request, idempotency_key: str = Header(alias="Idempotency-Key")):
    domain_id = await _require(request, "km_asset:use")
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    conversation = await _km_conversation(request, conversation_id, domain_id)
    spec = await _client(request).execution_spec(agent_id=UUID(str(conversation["agent_id"])), domain_id=domain_id, auth_context=request.state.auth_context)
    resource_context = spec.get("resource_context", {})
    fixed_collections = tuple(resource_context.get("collection_ids") or ())
    effective_level = await _effective_security_level(request)
    body = payload.model_dump(mode="json")
    body["execution_spec"] = spec
    body["collection_ids"] = list(fixed_collections)
    # KM Asset 统一按内部级别 1 入库。最终等级还必须受当前用户上限
    # 约束，浏览器只能主动降低本次检索范围，不能扩大授权范围。
    body["security_level"] = effective_level
    receipt = await runtime.create_conversation_turn(
        conversation_id=conversation_id,
        payload=body,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _km_turn_receipt(receipt)


@router.get("/conversations/{conversation_id}/turns")
async def list_conversation_turns(conversation_id: UUID, request: Request, after: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=500)):
    domain_id = await _require(request, "km_asset:use")
    await _km_conversation(request, conversation_id, domain_id)
    return await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).list_conversation_turns(conversation_id=conversation_id, after=after, limit=limit, auth_context=request.state.auth_context)
