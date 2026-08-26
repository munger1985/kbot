"""AIOps App 的成员、私有 Agent、连续对话和报告模板 BFF。"""

from typing import Any, Literal, cast
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.api.models import ModelCatalogItem, load_model_catalog
from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    UserAuthService,
    require_app_api_agent,
    require_app_api_permission,
    require_app_api_scope,
)
from platform_clients import KnowledgeCoreClient
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import PUBLIC_API_V1, PrincipalKind
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/aiops",
    tags=["AIOps App"],
)
AIOPS_PORTAL_DOMAIN_NAME = "aiops_portal"
AIOPS_MANUAL_COLLECTION_NAME = "operations-manuals"


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIOpsLoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class AIOpsPasswordChangePayload(_Payload):
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


class AIOpsCollectionModelsPayload(_Payload):
    parser_vlm: UUID | None = None
    embedding: UUID
    visual_embedding: UUID | None = None
    expected_row_version: int = Field(ge=1)


class AIOpsCollectionStatusPayload(_Payload):
    status: Literal["ACTIVE", "DISABLED"]


class AIOpsManualApprovalPayload(_Payload):
    comment: str | None = Field(default=None, max_length=1000)


class AIOpsAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_ids: tuple[UUID, ...] = Field(min_length=1, max_length=16)
    target_id: UUID | None = None
    allow_change_execution: bool = False
    auto_alert_enabled: bool = True
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] = "CRITICAL"
    alert_cooldown_minutes: int = Field(default=15, ge=0, le=1440)
    models: dict[str, UUID] = Field(default_factory=dict)
    image_capabilities: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AIOpsAgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=16
    )
    target_id: UUID | None = None
    allow_change_execution: bool | None = None
    auto_alert_enabled: bool | None = None
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] | None = None
    alert_cooldown_minutes: int | None = Field(default=None, ge=0, le=1440)
    models: dict[str, UUID] | None = None
    image_capabilities: dict[str, Any] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


class AIOpsAgentGrantPayload(_Payload):
    agent_id: UUID
    subject_type: Literal["USER", "ROLE"]
    subject_id: str = Field(min_length=1, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"


class AIOpsAgentGrantStatusPayload(_Payload):
    status: Literal["ACTIVE", "DISABLED"]
    expected_row_version: int = Field(ge=1)


class ConversationMessagePayload(_Payload):
    agent_id: UUID
    message: str = Field(min_length=1, max_length=32000)
    conversation_id: UUID | None = None
    source_run_id: UUID | None = None
    request_report: bool = False


class EvidenceRequestPayload(_Payload):
    purpose: str = Field(min_length=1, max_length=4000)
    suggested_sql: str | None = Field(default=None, max_length=32000)


class EvidenceTextPayload(_Payload):
    text: str = Field(min_length=1, max_length=32000)


class EvidenceUploadPayload(_Payload):
    filename: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=3, max_length=128)
    content_base64: str = Field(min_length=1, max_length=14_000_000)
    text: str | None = Field(default=None, max_length=32000)


class ReportTemplateCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    definition: dict[str, Any]


class ReportTemplateVersionPayload(_Payload):
    expected_row_version: int = Field(ge=1)
    definition: dict[str, Any]


def _domain_actor(request: Request) -> tuple[int, str]:
    context = get_auth_context(request)
    if context.app_id and context.app_id != "aiops":
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH"})
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"}) from exc
    if domain_id < 1:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, context.asserted_user_id or context.client_id


def _access(request: Request) -> AccessControlService:
    return cast(AccessControlService, request.app.state.access_control_service)


def _client(request: Request) -> AIOpsManagementClient:
    return cast(AIOpsManagementClient, request.app.state.aiops_client)


def _knowledge_client(request: Request) -> KnowledgeCoreClient:
    return cast(KnowledgeCoreClient, request.app.state.knowledge_core_client)


async def _fixed_manual_collection(
    request: Request, *, require_active: bool = False
) -> tuple[int, dict[str, Any]]:
    """取得当前 AIOps Domain 唯一的固定运维手册 Collection。"""
    domain_id, _ = _domain_actor(request)
    catalog = await _knowledge_client(request).list_collections(
        domain_id=domain_id,
        auth_context=request.state.auth_context,
    )
    matches = [
        item for item in catalog.get("collections", [])
        if item.get("display_name") == AIOPS_MANUAL_COLLECTION_NAME
    ]
    if len(matches) != 1:
        code = "AIOPS_KC_DUPLICATED" if matches else "AIOPS_KC_UNAVAILABLE"
        message = (
            "AIOps 固定运维手册 Collection 存在重复"
            if matches else "AIOps 固定运维手册 Collection 尚未初始化"
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT if matches else status.HTTP_503_SERVICE_UNAVAILABLE,
            {"code": code, "message": message},
        )
    collection = matches[0]
    metadata = collection.get("metadata") or {}
    if (
        metadata.get("owner_app_id") != "aiops"
        or metadata.get("fixed_resource") is not True
    ):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {
                "code": "AIOPS_KC_SCOPE_INVALID",
                "message": "AIOps 运维手册 Collection 的固定资源标识无效",
            },
        )
    if require_active and collection.get("status") != "ACTIVE":
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"code": "AIOPS_KC_DISABLED", "message": "AIOps 固定运维手册 Collection 未启用"},
        )
    return domain_id, collection


async def _validated_aiops_models(
    request: Request, *, parser_vlm: UUID | None,
    embedding: UUID, visual_embedding: UUID | None,
) -> dict[str, str]:
    """按平台模型目录校验 AIOps KC 模型类别。"""
    rows = await load_model_catalog(request)
    by_id = {str(item.get("model_id")): item for item in rows}
    requested = {"embedding": (embedding, 2)}
    if parser_vlm is not None:
        requested["parser_vlm"] = (parser_vlm, 5)
    if visual_embedding is not None:
        requested["visual_embedding"] = (visual_embedding, 3)
    result: dict[str, str] = {}
    for role, (model_id, category) in requested.items():
        row = by_id.get(str(model_id))
        if row is None or int(row.get("category") or 0) != category:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                {"code": "AIOPS_KC_MODEL_INVALID", "message": f"模型角色 {role} 不可用或类别不正确"},
            )
        result[role] = str(model_id)
    return result


async def _require(request: Request, permission: str):
    require_app_api_permission(request, permission)
    domain_id, actor_id = _domain_actor(request)
    try:
        snapshot = await _access(request).require(
            app_id="aiops",
            domain_id=domain_id,
            user_id=actor_id,
            permission_code=permission,
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403, {"code": "APP_PERMISSION_DENIED", "permission": permission}
        ) from exc
    return domain_id, actor_id, snapshot


async def _authorize_agent(request: Request, agent_id: UUID, snapshot, actor_id: str):
    require_app_api_agent(request, agent_id)
    if "aiops:agent_manage" in snapshot.permissions:
        return
    await _client(request).authorize_private_agent(
        {
            "agent_id": str(agent_id),
            "user_id": actor_id,
            "role_codes": list(snapshot.roles),
        },
        auth_context=request.state.auth_context,
    )


@router.post("/auth/login")
async def login(payload: AIOpsLoginPayload, request: Request):
    """使用平台用户凭据进入固定的 AIOps Portal Domain。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.login_for_domain_name(
        user_id=payload.user_id.strip(), password=payload.password,
        domain_name=AIOPS_PORTAL_DOMAIN_NAME, app_id="aiops",
    )


@router.post("/auth/password")
async def change_password(payload: AIOpsPasswordChangePayload, request: Request):
    """修改 AIOps 本地用户密码。"""
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.change_password(
        claims=request.state.user_token_claims,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_aiops_model_catalog(request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await load_model_catalog(request)


@router.get("/knowledge-core")
async def get_aiops_knowledge_core(request: Request):
    await _require(request, "aiops:knowledge_manage")
    domain_id, collection = await _fixed_manual_collection(request)
    policy = await _knowledge_client(request).get_collection_model_policy(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        auth_context=request.state.auth_context,
    )
    return {
        "collection_name": AIOPS_MANUAL_COLLECTION_NAME,
        "collection": collection,
        "model_policy": policy,
    }


@router.put("/knowledge-core/models")
async def update_aiops_knowledge_core_models(
    payload: AIOpsCollectionModelsPayload, request: Request,
):
    await _require(request, "aiops:knowledge_manage")
    domain_id, collection = await _fixed_manual_collection(request)
    models = await _validated_aiops_models(
        request, parser_vlm=payload.parser_vlm,
        embedding=payload.embedding,
        visual_embedding=payload.visual_embedding,
    )
    return await _knowledge_client(request).update_collection_models(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        payload={"models": models, "expected_row_version": payload.expected_row_version},
        auth_context=request.state.auth_context,
    )


@router.patch("/knowledge-core/status")
async def change_aiops_knowledge_core_status(
    payload: AIOpsCollectionStatusPayload, request: Request,
):
    await _require(request, "aiops:knowledge_manage")
    domain_id, collection = await _fixed_manual_collection(request)
    return await _knowledge_client(request).change_collection_status(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        status=payload.status,
        auth_context=request.state.auth_context,
    )


@router.post("/knowledge-core/manuals", status_code=status.HTTP_202_ACCEPTED)
async def upload_aiops_manual(request: Request):
    """把运维手册流式送入固定 KC，不在 Main API 落盘。"""
    await _require(request, "aiops:knowledge_manage")
    domain_id, collection = await _fixed_manual_collection(request, require_active=True)
    content_type = request.headers.get("Content-Type", "")
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()
    if not content_type.lower().startswith("multipart/form-data") or not idempotency_key:
        raise HTTPException(
            status.HTTP_428_PRECONDITION_REQUIRED,
            {
                "code": "AIOPS_KC_UPLOAD_HEADERS_REQUIRED",
                "message": "缺少 multipart Content-Type 或 Idempotency-Key",
            },
        )
    upstream = await _knowledge_client(request).ingest_multipart(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        intake_kind="user-files", content_type=content_type,
        body=request.stream(), idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return JSONResponse(status_code=upstream.status_code, content=upstream.payload)


@router.post("/knowledge-core/manuals/{bundle_revision_id}/approve")
async def approve_aiops_manual(
    bundle_revision_id: UUID,
    payload: AIOpsManualApprovalPayload,
    request: Request,
):
    await _require(request, "aiops:knowledge_manage")
    domain_id, collection = await _fixed_manual_collection(request)
    return await _knowledge_client(request).review_user_intake(
        domain_id=domain_id,
        collection_id=UUID(str(collection["collection_id"])),
        bundle_revision_id=bundle_revision_id,
        decision="APPROVE", comment=payload.comment,
        auth_context=request.state.auth_context,
    )


@router.get("/access")
async def get_access(request: Request):
    domain_id, actor_id = _domain_actor(request)
    snapshot = await _access(request).snapshot(
        app_id="aiops", domain_id=domain_id, user_id=actor_id
    )
    return {
        "app_id": snapshot.app_id,
        "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id,
        "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
    }


@router.get("/agents")
async def list_agents(request: Request):
    _, actor_id, snapshot = await _require(request, "aiops:use")
    agents = await _client(request).list_private_agents(
        auth_context=request.state.auth_context
    )
    require_app_api_scope(request, "aiops:agent:read")
    if request.state.auth_context.principal_kind == PrincipalKind.APP_API_CLIENT:
        allowed = {
            str(value)
            for value in request.state.auth_context.authorized_agent_ids
        }
        return [
            item for item in agents
            if item.get("status") == "ACTIVE"
            and str(item.get("agent_id")) in allowed
        ]
    if "aiops:agent_manage" in snapshot.permissions:
        return agents
    grants = await _client(request).list_private_agent_grants(
        auth_context=request.state.auth_context
    )
    allowed = {
        str(item["agent_id"])
        for item in grants
        if item.get("status") == "ACTIVE"
        and (
            (item.get("subject_type") == "USER" and item.get("subject_id") == actor_id)
            or (
                item.get("subject_type") == "ROLE"
                and item.get("subject_id") in snapshot.roles
            )
        )
    }
    return [
        item for item in agents
        if item.get("status") == "ACTIVE" and str(item.get("agent_id")) in allowed
    ]


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AIOpsAgentCreatePayload, request: Request):
    await _require(request, "aiops:agent_manage")
    return await _client(request).create_private_agent(
        payload.model_dump(mode="json"), auth_context=request.state.auth_context
    )


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    _, actor_id, snapshot = await _require(request, "aiops:use")
    require_app_api_scope(request, "aiops:agent:read")
    require_app_api_agent(request, agent_id)
    if "aiops:agent_manage" not in snapshot.permissions:
        await _authorize_agent(request, agent_id, snapshot, actor_id)
    agent = await _client(request).get_private_agent(
        agent_id, auth_context=request.state.auth_context
    )
    if (
        request.state.auth_context.principal_kind == PrincipalKind.APP_API_CLIENT
        and agent.get("status") != "ACTIVE"
    ):
        raise HTTPException(404, {"code": "AGENT_NOT_FOUND"})
    return agent


@router.patch("/agents/{agent_id}")
async def update_agent(
    agent_id: UUID, payload: AIOpsAgentUpdatePayload, request: Request
):
    await _require(request, "aiops:agent_manage")
    return await _client(request).update_private_agent(
        agent_id,
        payload.model_dump(mode="json", exclude_unset=True),
        auth_context=request.state.auth_context,
    )


@router.get("/agent-grants")
async def list_agent_grants(request: Request):
    await _require(request, "aiops:agent_manage")
    return await _client(request).list_private_agent_grants(
        auth_context=request.state.auth_context
    )


@router.put("/agent-grants")
async def upsert_agent_grant(payload: AIOpsAgentGrantPayload, request: Request):
    await _require(request, "aiops:agent_manage")
    return await _client(request).upsert_private_agent_grant(
        payload.model_dump(mode="json"), auth_context=request.state.auth_context
    )


@router.patch("/agent-grants/{grant_id}")
async def update_agent_grant(
    grant_id: UUID, payload: AIOpsAgentGrantStatusPayload, request: Request
):
    await _require(request, "aiops:agent_manage")
    return await _client(request).update_private_agent_grant(
        grant_id, payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/conversations", status_code=status.HTTP_201_CREATED)
async def create_or_append_conversation(
    payload: ConversationMessagePayload, request: Request
):
    _, actor_id, snapshot = await _require(request, "aiops:use")
    require_app_api_scope(request, "aiops:chat:write")
    await _authorize_agent(request, payload.agent_id, snapshot, actor_id)
    return await _client(request).conversation_request(
        "POST", "", payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/conversations")
async def list_conversations(
    request: Request,
    agent_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=50),
):
    _, actor_id, snapshot = await _require(request, "aiops:use")
    require_app_api_scope(request, "aiops:conversation:read")
    if agent_id is not None:
        await _authorize_agent(request, agent_id, snapshot, actor_id)
    query = {"limit": str(limit)}
    if agent_id is not None:
        query["agent_id"] = str(agent_id)
    rows = await _client(request).conversation_request(
        "GET", f"?{urlencode(query)}", auth_context=request.state.auth_context
    )
    if request.state.auth_context.principal_kind == PrincipalKind.APP_API_CLIENT:
        allowed = {
            str(value)
            for value in request.state.auth_context.authorized_agent_ids
        }
        return [
            item for item in rows
            if str(item.get("agent_id")) in allowed
        ]
    return rows


async def _conversation_with_access(
    request: Request, conversation_id: UUID
) -> tuple[dict[str, Any], Any, str]:
    _, actor_id, snapshot = await _require(request, "aiops:use")
    require_app_api_scope(request, "aiops:conversation:read")
    conversation = await _client(request).conversation_request(
        "GET", f"/{conversation_id}", auth_context=request.state.auth_context
    )
    await _authorize_agent(
        request, UUID(str(conversation["agent_id"])), snapshot, actor_id
    )
    return conversation, snapshot, actor_id


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: UUID, request: Request):
    conversation, _, _ = await _conversation_with_access(request, conversation_id)
    return conversation


@router.post("/conversations/{conversation_id}/evidence-requests", status_code=201)
async def request_evidence(
    conversation_id: UUID, payload: EvidenceRequestPayload, request: Request
):
    require_app_api_scope(request, "aiops:chat:write")
    await _conversation_with_access(request, conversation_id)
    return await _client(request).conversation_request(
        "POST", f"/{conversation_id}/evidence-requests",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/conversations/{conversation_id}/evidence-requests/{request_id}/text")
async def submit_evidence_text(
    conversation_id: UUID, request_id: UUID,
    payload: EvidenceTextPayload, request: Request,
):
    require_app_api_scope(request, "aiops:chat:write")
    await _conversation_with_access(request, conversation_id)
    return await _client(request).conversation_request(
        "POST", f"/{conversation_id}/evidence-requests/{request_id}/text",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/conversations/{conversation_id}/evidence-requests/{request_id}/skip")
async def skip_evidence(
    conversation_id: UUID, request_id: UUID,
    payload: EvidenceTextPayload, request: Request,
):
    require_app_api_scope(request, "aiops:chat:write")
    await _conversation_with_access(request, conversation_id)
    return await _client(request).conversation_request(
        "POST", f"/{conversation_id}/evidence-requests/{request_id}/skip",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/conversations/{conversation_id}/evidence-requests/{request_id}/uploads")
async def upload_evidence(
    conversation_id: UUID, request_id: UUID,
    payload: EvidenceUploadPayload, request: Request,
):
    require_app_api_scope(request, "aiops:chat:write")
    await _conversation_with_access(request, conversation_id)
    return await _client(request).conversation_request(
        "POST", f"/{conversation_id}/evidence-requests/{request_id}/uploads",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/report-templates")
async def list_report_templates(request: Request):
    await _require(request, "aiops:plan_manage")
    return await _client(request).report_template_request(
        "GET", "", auth_context=request.state.auth_context
    )


@router.get("/report-templates/{template_id}")
async def get_report_template(template_id: UUID, request: Request):
    await _require(request, "aiops:plan_manage")
    return await _client(request).report_template_request(
        "GET", f"/{template_id}", auth_context=request.state.auth_context
    )


@router.post("/report-templates", status_code=201)
async def create_report_template(
    payload: ReportTemplateCreatePayload, request: Request
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).report_template_request(
        "POST", "", payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/report-templates/{template_id}/versions", status_code=201)
async def create_report_template_version(
    template_id: UUID, payload: ReportTemplateVersionPayload, request: Request
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).report_template_request(
        "POST", f"/{template_id}/versions",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


__all__ = ["router"]
