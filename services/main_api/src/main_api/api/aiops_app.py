"""AIOps App 的成员、私有 Agent、连续对话和报告模板 BFF。"""

from typing import Any, Literal, cast
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    require_app_api_agent,
    require_app_api_permission,
    require_app_api_scope,
)
from platform_core.contracts import PrincipalKind
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/aiops",
    tags=["AIOps App"],
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AIOpsAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_id: UUID
    policy_id: UUID
    target_id: UUID | None = None
    inspection_plan_id: UUID | None = None
    models: dict[str, UUID] = Field(default_factory=dict)
    image_capabilities: dict[str, Any] = Field(default_factory=dict)
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] = Field(default_factory=dict)
    status: Literal["DRAFT", "ACTIVE"] = "DRAFT"


class AIOpsAgentUpdatePayload(_Payload):
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    diagnostic_source_id: UUID | None = None
    policy_id: UUID | None = None
    target_id: UUID | None = None
    inspection_plan_id: UUID | None = None
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
