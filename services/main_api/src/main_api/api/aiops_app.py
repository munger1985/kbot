"""AIOps App 的成员、私有 Agent、连续对话和报告模板 BFF。"""

import asyncio
import json
from datetime import datetime
from typing import Annotated, Any, Literal, cast
from urllib.parse import unquote
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from main_api.application import (
    UserAuthService,
    authorize_app_request,
    require_app_api_agent,
)
from platform_core.authorization import can_read_agent, filter_readable_agents
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import PUBLIC_API_V1, PrincipalKind
from platform_core.contracts.aiops import (
    ConversationUploadReceipt,
    ConversationStarterSelection,
    ConversationStarterCatalogView,
    ConversationSummary,
    InputContent,
    ReportArtifactView,
    ReportWindow,
    TargetFactConfirmCommand,
    TargetFactView,
    TurnReceipt,
    TurnSummary,
    TurnView,
    WorkloadDiffRequest,
)
from platform_core.contracts.aiops.workload import PGBADGER_MAX_UPLOAD_BYTES


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/aiops",
    tags=["AIOps App"],
)
AIOPS_PORTAL_DOMAIN_NAME = "aiops_portal"
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key")]


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


class OperationsKnowledgeReviewPayload(_Payload):
    expected_row_version: int = Field(ge=1)
    comment: str | None = Field(default=None, max_length=2000)


class OperationsKnowledgeSearchPayload(_Payload):
    query: str = Field(min_length=1, max_length=4000)
    purpose: Literal["DIAGNOSE", "EXPLAIN", "PLAN", "CHANGE", "VERIFY"] = "DIAGNOSE"
    problem_class: str | None = Field(default=None, max_length=128)
    components: tuple[str, ...] = Field(default=(), max_length=16)
    error_codes: tuple[str, ...] = Field(default=(), max_length=16)
    signal_names: tuple[str, ...] = Field(default=(), max_length=16)
    database_type: str | None = Field(default=None, max_length=32)
    database_major_version: str | None = Field(default=None, max_length=32)
    topology: str | None = Field(default=None, max_length=64)
    source_kinds: tuple[Literal["MANUAL", "DIAGNOSIS_CASE"], ...] = (
        "MANUAL", "DIAGNOSIS_CASE"
    )
    max_results: int = Field(default=8, ge=1, le=20)
    max_security_level: int = Field(default=3, ge=1, le=5)


class AIOpsControlledDynamicParameterRule(_Payload):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    allowed_values: tuple[str, ...] = Field(min_length=1, max_length=32)


class AIOpsControlledActionObjectScopes(_Payload):
    schemas: tuple[str, ...] = ()
    exclude_system_objects: bool = True
    dynamic_parameters: tuple[AIOpsControlledDynamicParameterRule, ...] = ()
    resource_manager_plans: tuple[str, ...] = ()
    privilege_grantees: tuple[str, ...] = ()
    system_privileges: tuple[str, ...] = ()
    object_privileges: tuple[str, ...] = ()


class AIOpsTargetControlledActionExecution(_Payload):
    target_id: UUID
    enabled: bool = False
    allowed_action_ids: tuple[str, ...] = ()
    object_scopes: AIOpsControlledActionObjectScopes = Field(
        default_factory=AIOpsControlledActionObjectScopes
    )
    max_daily_executions: int | None = Field(default=None, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_selection(self):
        if self.enabled != bool(self.allowed_action_ids):
            raise ValueError("启用受控动作时必须明确选择至少一个动作")
        if len(set(self.allowed_action_ids)) != len(self.allowed_action_ids):
            raise ValueError("受控动作不能重复")
        return self


class AIOpsAgentCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    target_ids: tuple[UUID, ...] = Field(min_length=1, max_length=32)
    controlled_action_execution: tuple[
        AIOpsTargetControlledActionExecution, ...
    ] = ()
    auto_alert_enabled: bool = True
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] = "CRITICAL"
    auto_observe_min_target_level: int = Field(default=1, ge=1, le=5)
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
    target_ids: tuple[UUID, ...] | None = Field(
        default=None, min_length=1, max_length=32
    )
    controlled_action_execution: tuple[
        AIOpsTargetControlledActionExecution, ...
    ] | None = None
    auto_alert_enabled: bool | None = None
    auto_observe_min_severity: Literal[
        "INFO", "WARNING", "HIGH", "CRITICAL"
    ] | None = None
    auto_observe_min_target_level: int | None = Field(
        default=None, ge=1, le=5
    )
    alert_cooldown_minutes: int | None = Field(default=None, ge=0, le=1440)
    models: dict[str, UUID] | None = None
    image_capabilities: dict[str, Any] | None = None
    instruction: str | None = Field(default=None, max_length=32000)
    config: dict[str, Any] | None = None
    status: Literal["DRAFT", "ACTIVE", "DISABLED", "ARCHIVED"] | None = None


class ConversationStartPayload(_Payload):
    agent_id: UUID
    target_id: UUID
    content: list[InputContent] = Field(default_factory=list, max_length=16)
    starter: ConversationStarterSelection | None = None
    title: str | None = Field(default=None, min_length=1, max_length=256)
    source_situation_id: UUID | None = None
    source_run_id: UUID | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "ConversationStartPayload":
        if self.source_situation_id is not None and self.source_run_id is not None:
            raise ValueError("会话来源只能选择 Situation 或 Run 其中之一")
        if not self.content and self.starter is None:
            raise ValueError("会话必须包含内容或功能入口选择")
        return self


class AIOpsConversationTurnPayload(_Payload):
    content: list[InputContent] = Field(default_factory=list, max_length=16)
    source_run_id: UUID | None = None
    starter: ConversationStarterSelection | None = None

    @model_validator(mode="after")
    def validate_input(self) -> "AIOpsConversationTurnPayload":
        if not self.content and self.starter is None:
            raise ValueError("Turn 必须包含内容或功能入口选择")
        return self


class InspectionTemplateCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class InspectionTemplateVersionPayload(_Payload):
    expected_row_version: int = Field(ge=1)
    selected_check_ids: tuple[str, ...] = Field(min_length=1)


class SessionReportTemplateCreatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    definition: dict[str, Any]


class SessionReportTemplateVersionPayload(_Payload):
    expected_row_version: int = Field(ge=1)
    definition: dict[str, Any]


def _client(request: Request) -> AIOpsManagementClient:
    return cast(AIOpsManagementClient, request.app.state.aiops_client)


async def _require(request: Request, permission: str):
    return await authorize_app_request(
        request,
        app_id="aiops",
        permission=permission,
    )


_AIOPS_AGENT_USE_FIELDS = (
    "agent_id",
    "domain_id",
    "display_name",
    "description",
    "status",
    "agent_version_id",
    "version_no",
    "target_ids",
    "target_candidates",
    "image_capabilities",
)


def _aiops_agent_use_view(agent: dict[str, Any]) -> dict[str, Any]:
    """仅返回聊天选择和输入能力所需的 Agent 字段。"""

    return {
        field: agent[field]
        for field in _AIOPS_AGENT_USE_FIELDS
        if field in agent
    }


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


@router.get("/operations-knowledge/overview")
async def operations_knowledge_overview(request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).get_operations_knowledge_overview(
        auth_context=request.state.auth_context
    )


@router.get("/operations-knowledge/assets")
async def list_operations_knowledge_assets(
    request: Request,
    asset_kind: str | None = None,
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).list_operations_knowledge_assets(
        asset_kind=asset_kind, status=status_filter, limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get("/operations-knowledge/assets/{asset_id}")
async def get_operations_knowledge_asset(asset_id: UUID, request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).get_operations_knowledge_asset(
        asset_id, auth_context=request.state.auth_context
    )


@router.get("/operations-knowledge/versions/{asset_version_id}")
async def get_operations_knowledge_version(asset_version_id: UUID, request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).get_operations_knowledge_version(
        asset_version_id, auth_context=request.state.auth_context
    )


@router.get("/operations-knowledge/versions/{asset_version_id}/source")
async def download_operations_knowledge_source(asset_version_id: UUID, request: Request):
    await _require(request, "aiops:knowledge_manage")
    result = await _client(request).download_operations_knowledge_source(
        asset_version_id, auth_context=request.state.auth_context
    )
    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=result.headers,
    )


@router.post("/operations-knowledge/manuals", status_code=status.HTTP_202_ACCEPTED)
async def upload_operations_manual(request: Request):
    """把原始手册有界转发到 AIOps，由业务服务登记资产并驱动 KC。"""
    await _require(request, "aiops:knowledge_manage")
    required = {
        "file_name": request.headers.get("X-File-Name", "").strip(),
        "metadata": request.headers.get("X-Upload-Metadata", "").strip(),
        "sha256": request.headers.get("X-Content-SHA256", "").strip(),
        "idempotency": request.headers.get("Idempotency-Key", "").strip(),
    }
    if not all(required.values()):
        raise HTTPException(
            status.HTTP_428_PRECONDITION_REQUIRED,
            {"code": "AIOPS_KNOWLEDGE_UPLOAD_HEADERS_REQUIRED", "message": "缺少运维手册上传头"},
        )
    metadata = json.loads(unquote(required["metadata"]))
    return await _client(request).upload_operations_manual(
        file_name=unquote(required["file_name"]),
        media_type=request.headers.get("Content-Type", "application/octet-stream"),
        body=request.stream(), metadata=metadata,
        content_sha256=required["sha256"],
        idempotency_key=required["idempotency"],
        auth_context=request.state.auth_context,
    )


@router.post(
    "/operations-knowledge/assets/{asset_id}/versions",
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_operations_manual_version(asset_id: UUID, request: Request):
    """给既有手册资产登记新版本，发布前保留上一版本继续检索。"""
    await _require(request, "aiops:knowledge_manage")
    required = {
        "file_name": request.headers.get("X-File-Name", "").strip(),
        "metadata": request.headers.get("X-Upload-Metadata", "").strip(),
        "sha256": request.headers.get("X-Content-SHA256", "").strip(),
        "idempotency": request.headers.get("Idempotency-Key", "").strip(),
        "row_version": request.headers.get(
            "X-Expected-Asset-Row-Version", ""
        ).strip(),
    }
    if not all(required.values()):
        raise HTTPException(
            status.HTTP_428_PRECONDITION_REQUIRED,
            {"code": "AIOPS_KNOWLEDGE_UPLOAD_HEADERS_REQUIRED", "message": "缺少手册版本上传头"},
        )
    try:
        expected_row_version = int(required["row_version"])
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"code": "AIOPS_KNOWLEDGE_VERSION_CONFLICT", "message": "资产版本号无效"},
        ) from exc
    metadata = json.loads(unquote(required["metadata"]))
    return await _client(request).upload_operations_manual_version(
        asset_id=asset_id,
        expected_asset_row_version=expected_row_version,
        file_name=unquote(required["file_name"]),
        media_type=request.headers.get("Content-Type", "application/octet-stream"),
        body=request.stream(),
        metadata=metadata,
        content_sha256=required["sha256"],
        idempotency_key=required["idempotency"],
        auth_context=request.state.auth_context,
    )


@router.post("/operations-knowledge/reports/{report_id}:extract-case", status_code=202)
async def extract_operations_diagnosis_case(report_id: UUID, request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).extract_operations_diagnosis_case(
        report_id, auth_context=request.state.auth_context
    )


@router.post("/operations-knowledge/versions/{asset_version_id}:reconcile")
async def reconcile_operations_knowledge_version(asset_version_id: UUID, request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).reconcile_operations_knowledge_version(
        asset_version_id, auth_context=request.state.auth_context
    )


async def _review_operations_knowledge(
    *, asset_version_id: UUID, decision: str,
    payload: OperationsKnowledgeReviewPayload, request: Request,
):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).review_operations_knowledge_version(
        asset_version_id, decision=decision,
        expected_row_version=payload.expected_row_version,
        comment=payload.comment,
        auth_context=request.state.auth_context,
    )


@router.post("/operations-knowledge/versions/{asset_version_id}:publish")
async def publish_operations_knowledge_version(asset_version_id: UUID, payload: OperationsKnowledgeReviewPayload, request: Request):
    return await _review_operations_knowledge(asset_version_id=asset_version_id, decision="publish", payload=payload, request=request)


@router.post("/operations-knowledge/versions/{asset_version_id}:reject")
async def reject_operations_knowledge_version(asset_version_id: UUID, payload: OperationsKnowledgeReviewPayload, request: Request):
    return await _review_operations_knowledge(asset_version_id=asset_version_id, decision="reject", payload=payload, request=request)


@router.post("/operations-knowledge/versions/{asset_version_id}:retire")
async def retire_operations_knowledge_version(asset_version_id: UUID, payload: OperationsKnowledgeReviewPayload, request: Request):
    return await _review_operations_knowledge(asset_version_id=asset_version_id, decision="retire", payload=payload, request=request)


@router.post("/operations-knowledge/versions/{asset_version_id}:retry")
async def retry_operations_knowledge_version(asset_version_id: UUID, payload: OperationsKnowledgeReviewPayload, request: Request):
    return await _review_operations_knowledge(asset_version_id=asset_version_id, decision="retry", payload=payload, request=request)


@router.get("/operations-knowledge/reviews")
async def list_operations_knowledge_reviews(request: Request, limit: int = Query(100, ge=1, le=500)):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).list_operations_knowledge_reviews(
        limit=limit, auth_context=request.state.auth_context
    )


@router.post("/operations-knowledge/search-preview")
async def preview_operations_knowledge_search(payload: OperationsKnowledgeSearchPayload, request: Request):
    await _require(request, "aiops:knowledge_manage")
    return await _client(request).search_operations_knowledge(
        payload.model_dump(mode="json"), auth_context=request.state.auth_context
    )


@router.get("/access")
async def get_access(request: Request):
    _, _, snapshot = await authorize_app_request(
        request, app_id="aiops", permission="aiops:use"
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
    domain_id, _, snapshot = await _require(request, "aiops:use")
    agents = await _client(request).list_private_agents(
        auth_context=request.state.auth_context
    )
    visible = filter_readable_agents(
        app_id="aiops",
        domain_id=domain_id,
        principal_kind=request.state.auth_context.principal_kind,
        permissions=snapshot.permissions,
        authorized_agent_ids=request.state.auth_context.authorized_agent_ids,
        agents=agents,
    )
    if (
        request.state.auth_context.principal_kind != PrincipalKind.APP_API_CLIENT
        and "aiops:agent_manage" in snapshot.permissions
    ):
        return visible
    return [_aiops_agent_use_view(dict(item)) for item in visible]


@router.get("/action-catalog/{target_id}")
async def action_catalog(target_id: UUID, request: Request):
    await _require(request, "aiops:agent_manage")
    return await _client(request).get_private_action_catalog(
        target_id, auth_context=request.state.auth_context
    )


@router.post("/agents", status_code=status.HTTP_201_CREATED)
async def create_agent(payload: AIOpsAgentCreatePayload, request: Request):
    await _require(request, "aiops:agent_manage")
    return await _client(request).create_private_agent(
        payload.model_dump(mode="json"), auth_context=request.state.auth_context
    )


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: UUID, request: Request):
    domain_id, _, snapshot = await _require(request, "aiops:use")
    require_app_api_agent(request, agent_id)
    agent = await _client(request).get_private_agent(
        agent_id, auth_context=request.state.auth_context
    )
    if not can_read_agent(
        app_id="aiops",
        domain_id=domain_id,
        principal_kind=request.state.auth_context.principal_kind,
        permissions=snapshot.permissions,
        authorized_agent_ids=request.state.auth_context.authorized_agent_ids,
        agent=agent,
    ):
        raise HTTPException(404, {"code": "AGENT_NOT_FOUND"})
    if (
        request.state.auth_context.principal_kind != PrincipalKind.APP_API_CLIENT
        and "aiops:agent_manage" in snapshot.permissions
    ):
        return agent
    return _aiops_agent_use_view(agent)


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


@router.post(
    "/conversation-uploads",
    status_code=status.HTTP_201_CREATED,
    response_model=ConversationUploadReceipt,
)
async def upload_conversation_input(request: Request):
    """把浏览器原始文件流转发给 AIOps，不在 Main API 落盘。"""
    await _require(request, "aiops:use")
    file_name = unquote(request.headers.get("X-File-Name", "").strip())
    media_type = request.headers.get("Content-Type", "").strip()
    if not file_name or not media_type:
        raise HTTPException(
            422,
            {
                "code": "AIOPS_UPLOAD_METADATA_REQUIRED",
                "message": "上传必须提供 X-File-Name 和 Content-Type",
            },
        )
    return await _client(request).upload_conversation_input(
        file_name=file_name,
        media_type=media_type,
        body=request.stream(),
        auth_context=request.state.auth_context,
    )


def _workload_report_view(payload: dict[str, Any]) -> ReportArtifactView:
    view = ReportArtifactView.model_validate(payload)
    return view.model_copy(
        update={
            "download_url": (
                f"{PUBLIC_API_V1}/apps/aiops/artifacts/"
                f"{view.artifact_id}/content"
            )
        }
    )


async def _bounded_pgbadger_body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="pgBadger上传Content-Length无效",
            ) from exc
        if declared_size < 0:
            raise HTTPException(400, "pgBadger上传Content-Length无效")
        if declared_size > PGBADGER_MAX_UPLOAD_BYTES:
            raise HTTPException(413, "pgBadger上传文件超过20MiB")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > PGBADGER_MAX_UPLOAD_BYTES:
            raise HTTPException(413, "pgBadger上传文件超过20MiB")
    return bytes(body)


@router.post(
    "/reports/workload",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_workload_report(
    payload: ReportWindow,
    request: Request,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    await _require(request, "aiops:use")
    result = await _client(request).create_workload_report(
        payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _workload_report_view(result)


@router.post(
    "/reports/workload-diff",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_workload_diff_report(
    payload: WorkloadDiffRequest,
    request: Request,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    await _require(request, "aiops:use")
    result = await _client(request).create_workload_diff_report(
        payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _workload_report_view(result)


@router.post(
    "/reports/activity",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def create_activity_report(
    payload: ReportWindow,
    request: Request,
    idempotency_key: IdempotencyKey,
) -> ReportArtifactView:
    await _require(request, "aiops:use")
    result = await _client(request).create_activity_report(
        payload.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _workload_report_view(result)


@router.post(
    "/postgresql/pgbadger-artifacts",
    response_model=ReportArtifactView,
    status_code=status.HTTP_201_CREATED,
)
async def import_postgresql_pgbadger_artifact(
    request: Request,
    idempotency_key: IdempotencyKey,
    target_id: UUID = Query(),
    period_start: datetime = Query(),
    period_end: datetime = Query(),
    file_name: str = Header(alias="X-File-Name"),
) -> ReportArtifactView:
    await _require(request, "aiops:use")
    result = await _client(request).import_postgresql_pgbadger_artifact(
        target_id=target_id,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        file_name=unquote(file_name)[:256],
        media_type=request.headers.get(
            "content-type", "application/octet-stream"
        ).split(";", 1)[0],
        body=await _bounded_pgbadger_body(request),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _workload_report_view(result)


@router.get("/artifacts/{artifact_id}/content")
async def download_report_artifact(
    artifact_id: UUID,
    request: Request,
) -> Response:
    await _require(request, "aiops:use")
    result = await _client(request).download_report_artifact(
        artifact_id,
        auth_context=request.state.auth_context,
    )
    return Response(
        content=result.body,
        media_type=result.media_type,
        headers=result.headers,
    )


@router.post(
    "/conversations",
    status_code=status.HTTP_201_CREATED,
    response_model=TurnReceipt,
)
async def start_conversation(
    payload: ConversationStartPayload,
    request: Request,
    idempotency_key: IdempotencyKey,
):
    await _require(request, "aiops:use")
    require_app_api_agent(request, payload.agent_id)
    if payload.source_run_id is not None:
        source = {
            "source_type": "RUN",
            "run_id": str(payload.source_run_id),
        }
    elif payload.source_situation_id is not None:
        source = {
            "source_type": "SITUATION",
            "situation_id": str(payload.source_situation_id),
        }
    else:
        source = {"source_type": "CHAT"}
    return await _client(request).start_conversation(
        {
            "conversation": {
                "agent_id": str(payload.agent_id),
                "target_id": str(payload.target_id),
                "title": payload.title,
                "source": source,
            },
            "first_turn": {
                "content": [
                    item.model_dump(mode="json") for item in payload.content
                ],
                "idempotency_key": idempotency_key,
                "source_run_id": (
                    str(payload.source_run_id)
                    if payload.source_run_id
                    else None
                ),
                "starter": (
                    payload.starter.model_dump(mode="json")
                    if payload.starter is not None
                    else None
                ),
            },
        },
        auth_context=request.state.auth_context,
    )


@router.get(
    "/conversation-starters",
    response_model=ConversationStarterCatalogView,
)
async def list_conversation_starters(
    request: Request,
    target_id: UUID,
    agent_id: UUID,
):
    """返回当前 Agent 与 Target 可以执行的结构化功能入口。"""
    await _require(request, "aiops:use")
    require_app_api_agent(request, agent_id)
    return await _client(request).list_conversation_starters(
        agent_id=agent_id,
        target_id=target_id,
        auth_context=request.state.auth_context,
    )


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(
    request: Request,
    agent_id: UUID | None = None,
    target_id: UUID | None = None,
    limit: int = Query(50, ge=1, le=50),
):
    await _require(request, "aiops:use")
    if agent_id is not None:
        require_app_api_agent(request, agent_id)
    rows = await _client(request).list_conversations(
        agent_id=agent_id,
        target_id=target_id,
        limit=limit,
        auth_context=request.state.auth_context,
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
    conversation = await _client(request).get_conversation(
        conversation_id,
        auth_context=request.state.auth_context,
    )
    require_app_api_agent(request, UUID(str(conversation["agent_id"])))
    return conversation, snapshot, actor_id


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationSummary,
)
async def get_conversation(conversation_id: UUID, request: Request):
    conversation, _, _ = await _conversation_with_access(request, conversation_id)
    return conversation


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ConversationSummary,
)
async def archive_conversation(conversation_id: UUID, request: Request):
    await _conversation_with_access(request, conversation_id)
    return await _client(request).archive_conversation(
        conversation_id,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/conversations/{conversation_id}/turns",
    status_code=202,
    response_model=TurnReceipt,
)
async def create_conversation_turn(
    conversation_id: UUID,
    payload: AIOpsConversationTurnPayload,
    request: Request,
    idempotency_key: IdempotencyKey,
):
    await _conversation_with_access(request, conversation_id)
    return await _client(request).create_conversation_turn(
        conversation_id,
        {
            "content": [
                item.model_dump(mode="json") for item in payload.content
            ],
            "idempotency_key": idempotency_key,
            "source_run_id": (
                str(payload.source_run_id) if payload.source_run_id else None
            ),
            "starter": (
                payload.starter.model_dump(mode="json")
                if payload.starter is not None
                else None
            ),
        },
        auth_context=request.state.auth_context,
    )


@router.get(
    "/conversations/{conversation_id}/turns",
    response_model=list[TurnSummary],
)
async def list_conversation_turns(
    conversation_id: UUID,
    request: Request,
    after_turn_no: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    await _conversation_with_access(request, conversation_id)
    return await _client(request).list_conversation_turns(
        conversation_id,
        after_turn_no=after_turn_no,
        limit=limit,
        auth_context=request.state.auth_context,
    )


@router.get(
    "/conversations/{conversation_id}/turns/{turn_id}",
    response_model=TurnView,
)
async def get_conversation_turn(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
):
    await _conversation_with_access(request, conversation_id)
    return await _client(request).get_conversation_turn(
        conversation_id,
        turn_id,
        auth_context=request.state.auth_context,
    )


@router.get("/conversations/{conversation_id}/turns/{turn_id}/inputs/{item_no}/content")
async def download_conversation_input_image(
    conversation_id: UUID,
    turn_id: UUID,
    item_no: int,
    request: Request,
):
    """代理已固化的会话图片，不向浏览器公开 AIOps 本地存储地址。"""
    if item_no < 1:
        raise HTTPException(
            422,
            {"code": "AIOPS_INPUT_ITEM_INVALID", "message": "输入序号必须大于零"},
        )
    await _conversation_with_access(request, conversation_id)
    upstream = await _client(request).download_conversation_input_image(
        conversation_id,
        turn_id,
        item_no,
        auth_context=request.state.auth_context,
    )
    return Response(
        content=upstream.body,
        media_type=upstream.media_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/conversations/{conversation_id}/turns/{turn_id}/artifacts/{artifact_id}/content"
)
async def download_conversation_artifact(
    conversation_id: UUID,
    turn_id: UUID,
    artifact_id: UUID,
    request: Request,
):
    """按会话授权边界代理不可变Artifact。"""
    await _conversation_with_access(request, conversation_id)
    upstream = await _client(request).download_conversation_artifact(
        conversation_id,
        turn_id,
        artifact_id,
        auth_context=request.state.auth_context,
    )
    return Response(
        content=upstream.body,
        media_type=upstream.media_type,
        headers={
            "Content-Disposition": upstream.headers.get(
                "Content-Disposition",
                'attachment; filename="aiops-artifact.bin"',
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.pdf",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def download_implementation_runbook_pdf(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
):
    """代理数据库实施操作文档 PDF，不向浏览器暴露 Agent 内部接口。"""
    await _conversation_with_access(request, conversation_id)
    upstream = await _client(request).download_implementation_runbook_pdf(
        conversation_id,
        turn_id,
        auth_context=request.state.auth_context,
    )
    return Response(
        content=upstream.body,
        media_type="application/pdf",
        headers={
            "Content-Disposition": upstream.headers.get(
                "Content-Disposition",
                f'attachment; filename="database-implementation-{turn_id}.pdf"',
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.md",
    response_class=Response,
    responses={200: {"content": {"text/markdown": {}}}},
)
async def download_implementation_runbook_markdown(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
):
    """代理数据库实施操作文档 Markdown，不暴露 Agent 内部接口。"""
    await _conversation_with_access(request, conversation_id)
    upstream = await _client(request).download_implementation_runbook_markdown(
        conversation_id,
        turn_id,
        auth_context=request.state.auth_context,
    )
    return Response(
        content=upstream.body,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": upstream.headers.get(
                "Content-Disposition",
                f'attachment; filename="database-implementation-{turn_id}.md"',
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get(
    "/conversations/{conversation_id}/turns/{turn_id}/implementation-runbook.zip",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}}},
)
async def download_implementation_runbook_zip(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
):
    """代理数据库实施脚本包。"""
    await _conversation_with_access(request, conversation_id)
    upstream = await _client(request).download_implementation_runbook_zip(
        conversation_id, turn_id, auth_context=request.state.auth_context,
    )
    return Response(
        content=upstream.body,
        media_type="application/zip",
        headers={
            "Content-Disposition": upstream.headers.get(
                "Content-Disposition",
                f'attachment; filename="database-implementation-{turn_id}.zip"',
            ),
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post(
    "/conversations/{conversation_id}/turns/{turn_id}/cancel",
    response_model=TurnSummary,
)
async def cancel_conversation_turn(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
):
    await _conversation_with_access(request, conversation_id)
    return await _client(request).cancel_conversation_turn(
        conversation_id,
        turn_id,
        auth_context=request.state.auth_context,
    )


@router.post(
    "/conversations/{conversation_id}/turns/{turn_id}/target-facts:confirm",
    response_model=TargetFactView,
)
async def confirm_conversation_target_fact(
    conversation_id: UUID,
    turn_id: UUID,
    body: TargetFactConfirmCommand,
    request: Request,
):
    await _conversation_with_access(request, conversation_id)
    payload = await _client(request).confirm_conversation_target_fact(
        conversation_id,
        turn_id,
        body.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return TargetFactView.model_validate(payload)


@router.get("/conversations/{conversation_id}/turns/{turn_id}/events")
async def stream_conversation_turn_events(
    conversation_id: UUID,
    turn_id: UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    await _conversation_with_access(request, conversation_id)
    try:
        cursor = int(last_event_id or "0")
    except ValueError as exc:
        raise HTTPException(
            400,
            {
                "code": "AIOPS_TURN_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 必须是非负整数",
            },
        ) from exc
    if cursor < 0:
        raise HTTPException(
            400,
            {
                "code": "AIOPS_TURN_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 不能为负数",
            },
        )
    client = _client(request)
    context = request.state.auth_context

    async def generate():
        nonlocal cursor
        while not await request.is_disconnected():
            page = await client.list_conversation_turn_events(
                conversation_id,
                turn_id,
                after_sequence=cursor,
                limit=200,
                auth_context=context,
            )
            for event in page["events"]:
                cursor = int(event["sequence_no"])
                yield (
                    f"id: {cursor}\n"
                    f"event: {event['event_type']}\n"
                    f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                )
            if page.get("terminal"):
                yield (
                    "event: done\n"
                    f"data: {json.dumps({'sequence_no': cursor})}\n\n"
                )
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/inspection-templates")
async def list_inspection_templates(request: Request):
    await _require(request, "aiops:plan_manage")
    return await _client(request).inspection_template_request(
        "GET", "", auth_context=request.state.auth_context
    )


@router.get("/inspection-templates/{template_id}")
async def get_inspection_template(template_id: UUID, request: Request):
    await _require(request, "aiops:plan_manage")
    return await _client(request).inspection_template_request(
        "GET", f"/{template_id}", auth_context=request.state.auth_context
    )


@router.post("/inspection-templates", status_code=201)
async def create_inspection_template(
    payload: InspectionTemplateCreatePayload, request: Request
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).inspection_template_request(
        "POST", "", payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/inspection-templates/{template_id}/versions", status_code=201)
async def create_inspection_template_version(
    template_id: UUID,
    payload: InspectionTemplateVersionPayload,
    request: Request,
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).inspection_template_request(
        "POST", f"/{template_id}/versions",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/session-report-templates")
async def list_session_report_templates(request: Request):
    await _require(request, "aiops:use")
    return await _client(request).session_report_template_request(
        "GET", "", auth_context=request.state.auth_context
    )


@router.get("/session-report-templates/{template_id}")
async def get_session_report_template(template_id: UUID, request: Request):
    await _require(request, "aiops:plan_manage")
    return await _client(request).session_report_template_request(
        "GET", f"/{template_id}", auth_context=request.state.auth_context
    )


@router.post("/session-report-templates", status_code=201)
async def create_session_report_template(
    payload: SessionReportTemplateCreatePayload, request: Request
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).session_report_template_request(
        "POST", "", payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.post("/session-report-templates/{template_id}/versions", status_code=201)
async def create_session_report_template_version(
    template_id: UUID,
    payload: SessionReportTemplateVersionPayload,
    request: Request,
):
    await _require(request, "aiops:plan_manage")
    return await _client(request).session_report_template_request(
        "POST", f"/{template_id}/versions",
        payload=payload.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )


@router.get("/report-layouts")
async def list_report_layouts(request: Request):
    await _require(request, "aiops:use")
    return await _client(request).list_report_layouts(
        auth_context=request.state.auth_context
    )


__all__ = ["router"]
