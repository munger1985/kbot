"""KM Asset App 公开 BFF API。"""

import json
import re
from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from loguru import logger
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    AppApiKeyError,
    KM_PORTAL_DOMAIN_NAME,
    UserAuthService,
    require_app_api_agent,
    require_app_api_permission,
    require_app_api_scope,
)
from platform_clients import AgentRuntimeClient, KmAssetClient, KnowledgeCoreClient
from platform_core.contracts import (
    AuthContext,
    ConversationQueryImage,
    PrincipalKind,
    PUBLIC_API_V1,
    UpdateConversationRequest,
)
from main_api.api.runs import (
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

_MAX_MANIFEST_BYTES = 1024 * 1024


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


class KmCollectionCreatePayload(_Payload):
    parser_llm: UUID
    embedding: UUID
    visual_embedding: UUID | None = None
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)


class KmCollectionStatusPayload(_Payload):
    status: Literal["ACTIVE", "DISABLED"]


class KmCollectionModelsPayload(_Payload):
    parser_llm: UUID
    embedding: UUID
    visual_embedding: UUID | None = None
    expected_row_version: int = Field(ge=1)


class BatchReindexItemPayload(_Payload):
    km_asset_id: UUID
    expected_row_version: int = Field(ge=1)


class BatchReindexPayload(_Payload):
    items: list[BatchReindexItemPayload] = Field(min_length=1, max_length=100)


class AssetAttachmentPreview(_Payload):
    """Asset 引用下可由用户二次打开的附件。"""

    document_version_id: UUID
    name: str
    document_role: str
    mime_type: str
    preview_type: Literal["PDF", "IMAGE", "TEXT", "DOWNLOAD"]
    evidence_source: bool = False
    page_no: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    content_url: str


class AssetReferencePreview(_Payload):
    """KM 引用首先投影 Asset，附件只能作为下级资源出现。"""

    reference_type: Literal["ASSET"] = "ASSET"
    citation_label: str
    title: str
    revision_no: int = Field(ge=1)
    status: str
    approval_status: str
    is_current_revision: bool
    asset_fields: dict[str, Any] = Field(default_factory=dict)
    asset_content_available: bool
    attachments: tuple[AssetAttachmentPreview, ...] = ()


class AgentCreatePayload(_Payload):
    source_id: UUID
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    models: dict[str, UUID] = Field(default_factory=dict)
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
    require_app_api_permission(request, permission)
    domain_id, actor_id = _domain_actor(request)
    service = cast(AccessControlService, request.app.state.access_control_service)
    try:
        await service.require(app_id="km_asset", domain_id=domain_id, user_id=actor_id, permission_code=permission)
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": permission}) from exc
    return domain_id


def _client(request: Request) -> KmAssetClient:
    return cast(KmAssetClient, request.app.state.km_asset_client)


def _runtime_auth_context(request: Request) -> AuthContext:
    """将 App API Key 映射为其绑定用户的 Portal 运行语义。

    公开接口的权限、Scope 与 Agent 白名单仍使用原始 App API Key
    上下文校验；这里只负责让下游运行时与 KM Portal 使用同一用户
    身份、Domain 和会话可见性。公开 API Key 不会被转发到内部服务。
    """
    context = get_auth_context(request)
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return context
    return AuthContext(
        principal_kind=PrincipalKind.PORTAL,
        client_id="user-session",
        request_id=context.request_id,
        trace_id=context.trace_id,
        api_key_id=context.api_key_id,
        entry_kind=context.entry_kind,
        app_id=context.app_id,
        domain_id=context.domain_id,
        tenant_id=context.tenant_id,
        asserted_user_id=context.asserted_user_id,
        roles=context.roles,
        scopes=context.scopes,
        authorized_agent_ids=context.authorized_agent_ids,
        delegated_by=context.client_id,
    )


def _clean_manifest_value(value: object) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip().strip("*_:： \t\r\n")


def _manifest_asset_fields(content: str) -> dict[str, str]:
    """从 manifest.md 的白名单元数据中投影 Slack 所需字段。"""
    title_match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    source_match = re.search(r"(?m)^Source ID:\s*(.+?)\s*$", content)
    metadata: dict[str, Any] = {}
    marker = re.search(r"(?m)^## Source metadata\s*$", content)
    if marker is not None:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(
                content[marker.end() :].lstrip()
            )
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            metadata = {
                str(key).strip().lower(): value
                for key, value in parsed.items()
            }
    aliases = {
        "external_asset_id": "asset_id",
        "asset_id": "asset_id",
        "asset_title": "asset_title",
        "title": "asset_title",
        "solution_briefing": "solution_briefing",
        "description": "solution_briefing",
        "asset_details": "solution_briefing",
        "author_mail": "author_mail",
        "author": "author_mail",
        "create_time": "create_time",
        "publish_date": "create_time",
        "asset_date": "create_time",
    }
    fields = {
        target: cleaned
        for source, target in aliases.items()
        if (cleaned := _clean_manifest_value(metadata.get(source)))
    }
    if "asset_id" not in fields and source_match is not None:
        fields["asset_id"] = _clean_manifest_value(source_match.group(1))
    if "asset_title" not in fields and title_match is not None:
        fields["asset_title"] = _clean_manifest_value(title_match.group(1))
    return fields


async def _reference_manifest_fields(
    request: Request,
    *,
    reference: _DocumentReference,
    files: list[dict[str, Any]],
) -> dict[str, str]:
    """通过 Main API 内部可信上下文读取引用 Bundle 的 manifest。"""
    manifest = next(
        (
            item
            for item in files
            if str(item.get("document_role") or "").upper() == "MANIFEST"
            and str(item.get("declared_name") or "").lower() == "manifest.md"
            and bool(item.get("preview_available"))
            and item.get("document_version_id")
        ),
        None,
    )
    if manifest is None:
        return {}
    mime_type = str(
        manifest.get("detected_mime_type")
        or manifest.get("declared_mime_type")
        or ""
    ).split(";", 1)[0].strip().lower()
    byte_size = int(manifest.get("byte_size") or 0)
    if mime_type not in {"text/markdown", "text/plain"} or byte_size > _MAX_MANIFEST_BYTES:
        return {}
    response = await cast(
        KnowledgeCoreClient, request.app.state.knowledge_core_client
    ).stream_source_file(
        domain_id=int(get_auth_context(request).domain_id or "0"),
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        document_version_id=UUID(str(manifest["document_version_id"])),
        range_header=None,
        auth_context=_runtime_auth_context(request),
    )
    if response.status_code != 200:
        return {}
    body = bytearray()
    async for chunk in response.body:
        body.extend(chunk)
        if len(body) > _MAX_MANIFEST_BYTES:
            return {}
    try:
        return _manifest_asset_fields(bytes(body).decode("utf-8-sig"))
    except UnicodeDecodeError:
        return {}


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
    collection = await _fixed_collection(
        request,
        domain_id=domain_id,
        require_active=True,
    )
    return UUID(str(collection["collection_id"]))


async def _fixed_collection(
    request: Request,
    *,
    domain_id: int,
    require_active: bool = False,
) -> dict[str, Any]:
    """取得当前 Domain 唯一的 KM Portal 固定 Collection。"""
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
    ]
    if len(matches) > 1:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "code": "KM_FIXED_COLLECTION_DUPLICATED",
                "message": "KM Portal 存在多个 assets Collection，请先清理重复数据",
            },
        )
    if not matches:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "KM_FIXED_COLLECTION_UNAVAILABLE",
                "message": "KM 固定 Collection assets 尚未初始化",
            },
        )
    collection = matches[0]
    if require_active and collection.get("status") != "ACTIVE":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "KM_FIXED_COLLECTION_DISABLED",
                "message": "KM 固定 Collection assets 当前未启用",
            },
        )
    return collection


async def _optional_fixed_collection(
    request: Request,
    *,
    domain_id: int,
) -> dict[str, Any] | None:
    """查询固定 Collection；缺失时允许管理页面进入创建态。"""
    try:
        return await _fixed_collection(request, domain_id=domain_id)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        if detail.get("code") == "KM_FIXED_COLLECTION_UNAVAILABLE":
            return None
        raise


async def _validated_collection_models(
    request: Request,
    *,
    parser_llm: UUID,
    embedding: UUID,
    visual_embedding: UUID | None,
) -> dict[str, str]:
    """依据启用模型目录校验 KC 模型角色与类别。"""
    catalog = await load_model_catalog(request)
    by_id = {str(item.get("model_id")): item for item in catalog}
    requested = {
        "parser_llm": (parser_llm, 1),
        "embedding": (embedding, 2),
    }
    if visual_embedding is not None:
        requested["visual_embedding"] = (visual_embedding, 3)
    models: dict[str, str] = {}
    for role, (model_id, expected_category) in requested.items():
        row = by_id.get(str(model_id))
        if row is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {
                    "code": "KM_COLLECTION_MODEL_UNAVAILABLE",
                    "message": f"模型角色 {role} 绑定的模型未启用或不存在",
                },
            )
        if int(row.get("category") or 0) != expected_category:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {
                    "code": "KM_COLLECTION_MODEL_CATEGORY_INVALID",
                    "message": f"模型角色 {role} 的模型类别不正确",
                },
            )
        models[role] = str(model_id)
    return models


async def _km_conversation(request: Request, conversation_id: UUID, domain_id: int):
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    runtime_context = _runtime_auth_context(request)
    conversation = await runtime.get_conversation(
        conversation_id=conversation_id,
        auth_context=runtime_context,
    )
    agent_id = UUID(str(conversation["agent_id"]))
    require_app_api_agent(request, agent_id)
    await _client(request).get_agent(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=runtime_context,
    )
    return conversation


async def _km_run(request: Request, run_id: UUID, domain_id: int):
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    runtime_context = _runtime_auth_context(request)
    run = await runtime.get_run(run_id=run_id, auth_context=runtime_context)
    agent_id = UUID(str(run["agent_id"]))
    require_app_api_agent(request, agent_id)
    await _client(request).get_agent(
        agent_id=agent_id,
        domain_id=domain_id,
        auth_context=runtime_context,
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
    """在 KM Token 的访问边界内返回管理 Agent 与 KC 所需模型。"""
    domain_id, actor_id = _domain_actor(request)
    service = cast(
        AccessControlService,
        request.app.state.access_control_service,
    )
    permissions = (
        "km_asset:agent_manage",
        "km_asset:knowledge_manage",
    )
    for permission in permissions:
        try:
            require_app_api_permission(request, permission)
            await service.require(
                app_id="km_asset",
                domain_id=domain_id,
                user_id=actor_id,
                permission_code=permission,
            )
            break
        except (AccessDeniedError, AppApiKeyError):
            continue
    else:
        raise HTTPException(
            403,
            {
                "code": "APP_PERMISSION_DENIED",
                "permissions_any": list(permissions),
            },
        )
    return await load_model_catalog(request)


@router.get("/knowledge-core")
async def get_km_knowledge_core(request: Request):
    """返回当前 Domain 的 KM Portal 固定 Collection。"""
    domain_id = await _require(request, "km_asset:knowledge_manage")
    collection = await _optional_fixed_collection(
        request,
        domain_id=domain_id,
    )
    return {
        "collection_name": KM_ASSET_COLLECTION_NAME,
        "collection": collection,
    }


@router.post("/knowledge-core", status_code=status.HTTP_201_CREATED)
async def create_km_knowledge_core(
    payload: KmCollectionCreatePayload,
    request: Request,
):
    """创建当前 Domain 唯一的 KM Portal 固定 Collection。"""
    domain_id = await _require(request, "km_asset:knowledge_manage")
    existing = await _optional_fixed_collection(request, domain_id=domain_id)
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {
                "code": "KM_FIXED_COLLECTION_EXISTS",
                "message": "KM 固定 Collection assets 已存在",
            },
        )
    models = await _validated_collection_models(
        request,
        parser_llm=payload.parser_llm,
        embedding=payload.embedding,
        visual_embedding=payload.visual_embedding,
    )
    return await cast(
        KnowledgeCoreClient,
        request.app.state.knowledge_core_client,
    ).create_collection(
        domain_id=domain_id,
        payload={
            "display_name": KM_ASSET_COLLECTION_NAME,
            "description": (
                payload.description
                or "KM Portal Asset 文档固定 Collection"
            ),
            "models": models,
            "default_security_level": payload.default_security_level,
            "metadata": {
                "owner_app_id": "km_asset",
                "fixed_resource": True,
            },
        },
        auth_context=request.state.auth_context,
    )


@router.patch("/knowledge-core/status")
async def change_km_knowledge_core_status(
    payload: KmCollectionStatusPayload,
    request: Request,
):
    """启用或停用 KM Portal 固定 Collection。"""
    domain_id = await _require(request, "km_asset:knowledge_manage")
    collection = await _fixed_collection(request, domain_id=domain_id)
    return await cast(
        KnowledgeCoreClient,
        request.app.state.knowledge_core_client,
    ).change_collection_status(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        status=payload.status,
        auth_context=request.state.auth_context,
    )


@router.put("/knowledge-core/models")
async def update_km_knowledge_core_models(
    payload: KmCollectionModelsPayload,
    request: Request,
):
    """更新固定 Collection 模型并遵守 KC 的不可变模型约束。"""
    domain_id = await _require(request, "km_asset:knowledge_manage")
    collection = await _fixed_collection(request, domain_id=domain_id)
    models = await _validated_collection_models(
        request,
        parser_llm=payload.parser_llm,
        embedding=payload.embedding,
        visual_embedding=payload.visual_embedding,
    )
    return await cast(
        KnowledgeCoreClient,
        request.app.state.knowledge_core_client,
    ).update_collection_models(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        payload={
            "models": models,
            "expected_row_version": payload.expected_row_version,
        },
        auth_context=request.state.auth_context,
    )


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


@router.post("/assets/actions/reindex", status_code=status.HTTP_202_ACCEPTED)
async def batch_reindex_assets(payload: BatchReindexPayload, request: Request):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).batch_reindex_assets(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
    )


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


@router.get("/jobs/processing")
async def list_processing_jobs(request: Request, source_id: UUID | None = None, limit: int = Query(default=500, ge=1, le=2000)):
    domain_id = await _require(request, "km_asset:operations_manage")
    return await _client(request).list_processing_jobs(domain_id=domain_id, source_id=source_id, limit=limit, auth_context=request.state.auth_context)


@router.get("/agents")
async def list_agents(request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:agent:read")
    rows = await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)
    allowed = {str(value) for value in request.state.auth_context.authorized_agent_ids}
    if request.state.auth_context.principal_kind.value == "APP_API_CLIENT":
        return [
            item for item in rows
            if item.get("status") == "ACTIVE"
            and str(item.get("agent_id")) in allowed
        ]
    return rows


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AgentCreatePayload, request: Request):
    domain_id = await _require(request, "km_asset:agent_manage")
    return await _client(request).create_agent(payload={"domain_id": domain_id, **payload.model_dump(mode="json")}, auth_context=request.state.auth_context)


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:agent:read")
    require_app_api_agent(request, agent_id)
    agent = await _client(request).get_agent(agent_id=agent_id, domain_id=domain_id, auth_context=request.state.auth_context)
    if (
        request.state.auth_context.principal_kind.value == "APP_API_CLIENT"
        and agent.get("status") != "ACTIVE"
    ):
        raise HTTPException(404, {"code": "AGENT_NOT_FOUND"})
    return agent


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
    require_app_api_scope(request, "km:chat:write")
    require_app_api_agent(request, payload.agent_id)
    runtime_context = _runtime_auth_context(request)
    spec = await _client(request).execution_spec(
        agent_id=payload.agent_id,
        domain_id=domain_id,
        auth_context=runtime_context,
    )
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    return await runtime.create_conversation(
        payload={**payload.model_dump(mode="json"), "execution_spec": spec},
        auth_context=runtime_context,
    )


@router.get("/conversations")
async def list_conversations(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:conversation:read")
    agents = await _client(request).list_agents(domain_id=domain_id, auth_context=request.state.auth_context)
    agent_ids = {str(item["agent_id"]) for item in agents}
    if request.state.auth_context.principal_kind.value == "APP_API_CLIENT":
        agent_ids &= {
            str(value)
            for value in request.state.auth_context.authorized_agent_ids
        }
    rows = await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).list_conversations(limit=200, auth_context=request.state.auth_context)
    return [item for item in rows if str(item.get("agent_id")) in agent_ids][:limit]


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:conversation:read")
    return await _km_conversation(request, conversation_id, domain_id)


@router.patch("/conversations/{conversation_id}")
async def update_conversation(conversation_id: UUID, payload: UpdateConversationRequest, request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:conversation:update")
    await _km_conversation(request, conversation_id, domain_id)
    return await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).update_conversation(conversation_id=conversation_id, payload=payload.model_dump(mode="json"), auth_context=request.state.auth_context)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, request: Request, expected_row_version: int = Query(ge=1)):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:conversation:delete")
    await _km_conversation(request, conversation_id, domain_id)
    await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).delete_conversation(conversation_id=conversation_id, expected_row_version=expected_row_version, auth_context=request.state.auth_context)
    return Response(status_code=204)


@router.get("/runs/{run_id}")
async def get_run(run_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:run:read")
    return await _km_run(request, run_id, domain_id)


@router.get("/runs/{run_id}/result")
async def get_run_result(run_id: UUID, request: Request):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:run:read")
    await _km_run(request, run_id, domain_id)
    return await cast(
        AgentRuntimeClient, request.app.state.agent_runtime_client
    ).get_result(run_id=run_id, auth_context=_runtime_auth_context(request))


@router.get("/runs/{run_id}/events")
async def stream_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:run:read")
    cursor = _parse_cursor(last_event_id)
    summary = await _km_run(request, run_id, domain_id)
    if cursor > int(summary["event_cursor"]):
        raise HTTPException(400, {"code": "AGENT_EVENT_CURSOR_INVALID", "message": "Last-Event-ID 超过当前 Run 事件游标"})
    return StreamingResponse(
        _event_stream(run_id=run_id, request=request, cursor=cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _document_reference(
    request: Request, run_id: UUID, citation_label: str
) -> tuple[_DocumentReference, dict[str, Any]]:
    """读取已完成回答中的不可变引用和同一回答数据。"""
    await _km_run(request, run_id, int(request.state.auth_context.domain_id))
    artifact = await cast(
        AgentRuntimeClient, request.app.state.agent_runtime_client
    ).get_result(run_id=run_id, auth_context=_runtime_auth_context(request))
    payload = artifact.get("payload")
    references = payload.get("references") if isinstance(payload, dict) else None
    raw = next((item for item in references or [] if isinstance(item, dict) and item.get("reference_type") == "DOCUMENT" and item.get("citation_label") == citation_label), None)
    if raw is None:
        raise _reference_not_found()
    try:
        return _DocumentReference.model_validate(raw), payload
    except ValueError as exc:
        raise HTTPException(409, {"code": "DOCUMENT_REFERENCE_INVALID", "message": "Run 引用缺少不可变文档定位信息"}) from exc


def _asset_attachment(
    *,
    run_id: UUID,
    citation_label: str,
    item: dict[str, Any],
    evidence_document_version_id: UUID,
    locator: tuple[
        int | None,
        int | None,
        tuple[float, float, float, float] | None,
    ],
) -> AssetAttachmentPreview | None:
    """把 Bundle 成员投影为 Asset 下级附件，忽略 manifest。"""
    if (
        str(item.get("document_role") or "").upper() == "MANIFEST"
        or not bool(item.get("preview_available"))
        or not item.get("document_version_id")
    ):
        return None
    document_version_id = UUID(str(item["document_version_id"]))
    mime_type = str(
        item.get("detected_mime_type")
        or item.get("declared_mime_type")
        or "application/octet-stream"
    ).split(";", 1)[0].strip().lower()
    evidence_source = document_version_id == evidence_document_version_id
    page_no, page_end, _bbox = locator if evidence_source else (None, None, None)
    return AssetAttachmentPreview(
        document_version_id=document_version_id,
        name=str(
            item.get("declared_name")
            or item.get("external_document_id")
            or "附件"
        ),
        document_role=str(item.get("document_role") or "ATTACHMENT"),
        mime_type=mime_type,
        preview_type=_preview_type(mime_type),
        evidence_source=evidence_source,
        page_no=page_no,
        page_end=page_end,
        content_url=(
            f"{PUBLIC_API_V1}/apps/km-asset/runs/{run_id}/references/"
            f"{citation_label}/files/{document_version_id}/content"
        ),
    )


@router.get(
    "/runs/{run_id}/references/{citation_label}/preview",
    response_model=AssetReferencePreview,
)
async def get_reference_preview(run_id: UUID, citation_label: str, request: Request):
    await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:reference:read")
    reference, _payload = await _document_reference(
        request, run_id, citation_label
    )
    runtime_context = _runtime_auth_context(request)
    preview = await cast(
        KnowledgeCoreClient, request.app.state.knowledge_core_client
    ).get_bundle_revision_preview(
        domain_id=int(get_auth_context(request).domain_id or "0"),
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        auth_context=runtime_context,
    )
    files = [item for item in preview.get("files", []) if isinstance(item, dict)]
    try:
        manifest_fields = await _reference_manifest_fields(
            request,
            reference=reference,
            files=files,
        )
    except Exception as exc:
        # Manifest 是 Slack Template 唯一元数据来源。底层文件已迁移、
        # 暂不可读或历史引用不完整时只返回基础预览；Slack 会保留
        # KBot 原始回答，不使用 QueryResult 拼装伪 Template。
        logger.warning(
            "KM Asset 参考 Manifest 读取失败，返回基础预览："
            "run_id={} citation_label={} cause={}",
            run_id,
            citation_label,
            str(exc),
        )
        manifest_fields = {}
    locator = _document_locator(reference)
    attachments = tuple(
        attachment
        for item in files
        if (
            attachment := _asset_attachment(
                run_id=run_id,
                citation_label=reference.citation_label,
                item=item,
                evidence_document_version_id=reference.document_version_id,
                locator=locator,
            )
        ) is not None
    )
    return AssetReferencePreview(
        citation_label=reference.citation_label,
        title=str(preview.get("title") or reference.title),
        revision_no=int(preview.get("revision_no") or 1),
        status=str(preview.get("status") or "UNKNOWN"),
        approval_status=str(preview.get("approval_status") or "UNKNOWN"),
        is_current_revision=bool(preview.get("is_current_revision")),
        asset_fields=manifest_fields,
        asset_content_available=any(
            str(item.get("document_role") or "").upper() == "MANIFEST"
            and bool(item.get("preview_available"))
            for item in files
        ),
        attachments=attachments,
    )


@router.get(
    "/runs/{run_id}/references/{citation_label}/files/"
    "{document_version_id}/content"
)
async def stream_reference_attachment(
    run_id: UUID,
    citation_label: str,
    document_version_id: UUID,
    request: Request,
    range_header: str | None = Header(default=None, alias="Range"),
):
    """只允许打开当前 Asset 引用预览中列出的下级附件。"""
    await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:reference:read")
    reference, _payload = await _document_reference(
        request, run_id, citation_label
    )
    preview = await cast(
        KnowledgeCoreClient, request.app.state.knowledge_core_client
    ).get_bundle_revision_preview(
        domain_id=int(request.state.auth_context.domain_id),
        collection_id=reference.collection_id,
        bundle_id=reference.bundle_id,
        bundle_revision_id=reference.bundle_revision_id,
        auth_context=request.state.auth_context,
    )
    attachment = next(
        (
            item
            for item in preview.get("files", [])
            if isinstance(item, dict)
            and str(item.get("document_version_id"))
            == str(document_version_id)
            and str(item.get("document_role") or "").upper() != "MANIFEST"
            and bool(item.get("preview_available"))
        ),
        None,
    )
    if attachment is None:
        raise _reference_not_found()
    upstream = await cast(KnowledgeCoreClient, request.app.state.knowledge_core_client).stream_source_file(domain_id=int(request.state.auth_context.domain_id), collection_id=reference.collection_id, bundle_id=reference.bundle_id, bundle_revision_id=reference.bundle_revision_id, document_version_id=document_version_id, range_header=range_header, auth_context=request.state.auth_context)
    forwarded = {name: upstream.headers[name] for name in ("accept-ranges", "cache-control", "content-disposition", "content-length", "content-range", "content-security-policy", "x-content-type-options") if name in upstream.headers}
    return StreamingResponse(upstream.body, status_code=upstream.status_code, media_type=upstream.headers.get("content-type", "application/octet-stream"), headers=forwarded)


@router.post("/conversations/{conversation_id}/turns", status_code=status.HTTP_202_ACCEPTED)
async def create_conversation_turn(conversation_id: UUID, payload: ConversationTurnPayload, request: Request, idempotency_key: str = Header(alias="Idempotency-Key")):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:chat:write")
    runtime = cast(AgentRuntimeClient, request.app.state.agent_runtime_client)
    conversation = await _km_conversation(request, conversation_id, domain_id)
    runtime_context = _runtime_auth_context(request)
    spec = await _client(request).execution_spec(
        agent_id=UUID(str(conversation["agent_id"])),
        domain_id=domain_id,
        auth_context=runtime_context,
    )
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
        auth_context=runtime_context,
    )
    return _km_turn_receipt(receipt)


@router.get("/conversations/{conversation_id}/turns")
async def list_conversation_turns(conversation_id: UUID, request: Request, after: int = Query(default=0, ge=0), limit: int = Query(default=200, ge=1, le=500)):
    domain_id = await _require(request, "km_asset:use")
    require_app_api_scope(request, "km:conversation:read")
    await _km_conversation(request, conversation_id, domain_id)
    return await cast(AgentRuntimeClient, request.app.state.agent_runtime_client).list_conversation_turns(conversation_id=conversation_id, after=after, limit=limit, auth_context=request.state.auth_context)
