"""Portal 可见的 AIOps 配置管理接口。"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal, TypeVar, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts import PUBLIC_API_V1
from platform_core.contracts.aiops import (
    AgentBindingCreate,
    AgentBindingPatch,
    AgentBindingView,
    ApprovalCommand,
    ApprovalReceipt,
    CancelRunCommand,
    ConnectivityCheckReceipt,
    DiagnosticQueryApprovalDecision,
    HitlResponse,
    HitlResult,
    HitlSkipCommand,
    InspectionFirePage,
    InspectionFireView,
    InspectionPlanCreate,
    InspectionPlanDetail,
    InspectionPlanPage,
    InspectionPlanPatch,
    ManualResultCommand,
    ManualResultReceipt,
    NotificationSubscriptionList,
    NotificationSubscriptionUpsert,
    NotificationSubscriptionView,
    SourceBindingCreate,
    SourceBindingPatch,
    SourceBindingView,
    DiagnosticSourceCreate,
    DiagnosticSourceConnectionTestResult,
    DiagnosticSourceDetail,
    DiagnosticSourcePage,
    DiagnosticSourcePatch,
    OpsCommand,
    OpsRunCreate,
    OpsRunReceipt,
    OpsRunResult,
    OpsRunPage,
    OpsRunSummary,
    PendingInputView,
    PolicyCreate,
    PolicyDetail,
    PolicyPage,
    ProposalView,
    ProposalPage,
    SituationPage,
    SituationView,
    RejectionCommand,
    ReportPage,
    ReportVersionPage,
    ReportView,
    TargetCreate,
    TargetConnectionTest,
    TargetConnectionTestResult,
    DatabaseCredentialInput,
    TargetDetail,
    TargetPage,
    TargetPatch,
    WebhookKeyRotation,
)
from platform_core.contracts.aiops.internal import CreateOpsRunCommand
from platform_core.identity import uuid7
from platform_core.security import get_auth_context
from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    require_app_api_permission,
)


async def _require_route_access(request: Request) -> None:
    context = get_auth_context(request)
    domain_id = int(context.domain_id or "0")
    actor_id = context.asserted_user_id or context.client_id
    relative = request.url.path.removeprefix(f"{PUBLIC_API_V1}/apps/aiops")
    permission = "aiops:use"
    if relative.startswith("/targets"):
        permission = "aiops:target_manage"
    elif relative.startswith("/diagnostic-sources"):
        permission = "aiops:diagnostic_source_manage"
    elif relative.startswith("/policies"):
        permission = "aiops:policy_manage"
    elif relative.startswith("/inspection-plans"):
        permission = "aiops:plan_manage"
    elif relative.endswith(("/approve", "/reject", "/manual-result")):
        permission = "aiops:proposal:approve"
    require_app_api_permission(request, permission)
    service = cast(
        AccessControlService, request.app.state.access_control_service
    )
    try:
        await service.require(
            app_id="aiops", domain_id=domain_id, user_id=actor_id,
            permission_code=permission,
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403, {"code": "APP_PERMISSION_DENIED", "permission": permission}
        ) from exc


router = APIRouter(
    prefix=f"{PUBLIC_API_V1}/apps/aiops",
    tags=["AIOps"],
    dependencies=[Depends(_require_route_access)],
)
IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key")]
ModelT = TypeVar("ModelT", bound=BaseModel)


def require_if_match(
    value: str | None = Header(default=None, alias="If-Match"),
) -> str:
    if value is None:
        raise HTTPException(
            status_code=428,
            detail={
                "code": "PRECONDITION_REQUIRED",
                "message": "该操作必须提供 If-Match",
            },
        )
    return value


IfMatch = Annotated[str, Depends(require_if_match)]


def _client(request: Request) -> AIOpsManagementClient:
    return cast(AIOpsManagementClient, request.app.state.aiops_client)


def _validated(
    model_type: type[ModelT],
    payload: dict,
    response: Response | None = None,
) -> ModelT:
    result = model_type.model_validate(payload)
    row_version = getattr(result, "row_version", None)
    if response is not None and row_version is not None:
        response.headers["ETag"] = f'"rv-{int(row_version)}"'
    return result


@router.get(
    "/notification-subscriptions",
    response_model=NotificationSubscriptionList,
)
async def list_notification_subscriptions(
    request: Request,
) -> NotificationSubscriptionList:
    payload = await _client(request).list_notification_subscriptions(
        auth_context=request.state.auth_context
    )
    return NotificationSubscriptionList.model_validate(payload)


@router.put(
    "/notification-subscriptions/targets/{target_id}",
    response_model=NotificationSubscriptionView,
)
async def upsert_notification_subscription(
    target_id: UUID,
    body: NotificationSubscriptionUpsert,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
) -> NotificationSubscriptionView:
    payload = await _client(request).upsert_notification_subscription(
        target_id,
        body.model_dump(mode="json"),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(NotificationSubscriptionView, payload, response)


@router.delete(
    "/notification-subscriptions/targets/{target_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def disable_notification_subscription(
    target_id: UUID,
    request: Request,
    if_match: IfMatch,
) -> Response:
    await _client(request).disable_notification_subscription(
        target_id,
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/reports/{report_id}", response_model=ReportView)
async def get_report(
    report_id: UUID,
    request: Request,
) -> ReportView:
    payload = await _client(request).get_report(
        report_id,
        auth_context=request.state.auth_context,
    )
    return ReportView.model_validate(payload)


@router.get("/reports", response_model=ReportPage)
async def list_reports(
    request: Request,
    target_id: UUID | None = None,
    report_type: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> ReportPage:
    payload = await _client(request).list_reports(
        target_id=target_id,
        report_type=report_type,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return ReportPage.model_validate(payload)


@router.get(
    "/reports/{report_id}/versions",
    response_model=ReportVersionPage,
)
async def list_report_versions(
    report_id: UUID,
    request: Request,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> ReportVersionPage:
    payload = await _client(request).list_report_versions(
        report_id,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return ReportVersionPage.model_validate(payload)


@router.get("/inspection-fires", response_model=InspectionFirePage)
async def list_inspection_fires(
    request: Request,
    plan_id: UUID | None = None,
    status: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> InspectionFirePage:
    payload = await _client(request).list_inspection_fires(
        plan_id=plan_id,
        status=status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return InspectionFirePage.model_validate(payload)


@router.get(
    "/inspection-fires/{fire_id}",
    response_model=InspectionFireView,
)
async def get_inspection_fire(
    fire_id: UUID,
    request: Request,
) -> InspectionFireView:
    payload = await _client(request).get_inspection_fire(
        fire_id,
        auth_context=request.state.auth_context,
    )
    return InspectionFireView.model_validate(payload)


@router.post("/runs", response_model=OpsRunReceipt, status_code=201)
async def create_ops_run(
    body: OpsRunCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> OpsRunReceipt:
    context = request.state.auth_context
    access = cast(
        AccessControlService, request.app.state.access_control_service
    )
    actor_id = context.asserted_user_id or context.client_id
    snapshot = await access.snapshot(
        app_id="aiops", domain_id=int(context.domain_id), user_id=actor_id
    )
    await _client(request).authorize_private_agent(
        {
            "agent_id": str(body.agent_id),
            "user_id": actor_id,
            "role_codes": list(snapshot.roles),
        },
        auth_context=context,
    )
    command = CreateOpsRunCommand(
        command_id=uuid7(),
        idempotency_key=idempotency_key,
        domain_id=int(context.domain_id),
        actor_id=actor_id,
        agent_id=body.agent_id,
        target_id=body.target_id,
        trigger_type="CHAT",
        input=body.input,
        session_id=body.session_id,
        blueprint_id="diagnosis.root-cause",
        blueprint_version="1",
        observation_start=body.observation_start,
        observation_end=body.observation_end,
        client_metadata={
            **body.client_metadata,
            "trace_id": context.trace_id,
        },
    )
    payload = await _client(request).create_run(
        command, auth_context=context
    )
    result = OpsRunReceipt(
        **payload,
        events_url=(
            f"{PUBLIC_API_V1}/apps/aiops/runs/{payload['ops_run_id']}/events"
        ),
    )
    response.headers["ETag"] = f'"rv-{result.row_version}"'
    return result


@router.get("/runs", response_model=OpsRunPage)
async def list_ops_runs(
    request: Request, target_id: UUID | None = None,
    status: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> OpsRunPage:
    payload = await _client(request).list_runs(
        target_id=target_id, status=status, cursor=cursor, limit=limit,
        auth_context=request.state.auth_context,
    )
    return OpsRunPage.model_validate(payload)


@router.get("/situations", response_model=SituationPage)
async def list_situations(
    request: Request, target_id: UUID | None = None,
    status: str | None = None, severity: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> SituationPage:
    payload = await _client(request).list_situations(
        target_id=target_id, status=status, severity=severity,
        cursor=cursor, limit=limit, auth_context=request.state.auth_context,
    )
    return SituationPage.model_validate(payload)


@router.get("/situations/{situation_id}", response_model=SituationView)
async def get_situation(situation_id: UUID, request: Request) -> SituationView:
    payload = await _client(request).get_situation(
        situation_id, auth_context=request.state.auth_context
    )
    return SituationView.model_validate(payload)


@router.get("/runs/{run_id}", response_model=OpsRunSummary)
async def get_ops_run(
    run_id: UUID, request: Request, response: Response
) -> OpsRunSummary:
    payload = await _client(request).get_run(
        run_id, auth_context=request.state.auth_context
    )
    result = OpsRunSummary.model_validate(payload)
    response.headers["ETag"] = f'"rv-{result.row_version}"'
    return result


@router.get("/runs/{run_id}/result", response_model=OpsRunResult)
async def get_ops_run_result(
    run_id: UUID,
    request: Request,
) -> OpsRunResult:
    payload = await _client(request).get_run_result(
        run_id,
        auth_context=request.state.auth_context,
    )
    return OpsRunResult.model_validate(payload)


@router.get(
    "/runs/{run_id}/pending-input",
    response_model=PendingInputView,
)
async def get_pending_input(
    run_id: UUID,
    request: Request,
    response: Response,
) -> PendingInputView:
    payload = await _client(request).get_pending_input(
        run_id, auth_context=request.state.auth_context
    )
    return _validated(PendingInputView, payload, response)


@router.get("/hitl/{hitl_id}", response_model=PendingInputView)
async def get_hitl_input(
    hitl_id: UUID,
    request: Request,
    response: Response,
) -> PendingInputView:
    payload = await _client(request).get_hitl_input(
        hitl_id, auth_context=request.state.auth_context
    )
    return _validated(PendingInputView, payload, response)


@router.post("/hitl/{hitl_id}/response", response_model=HitlResult)
async def respond_hitl(
    hitl_id: UUID,
    body: HitlResponse,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> HitlResult:
    payload = await _client(request).respond_hitl(
        hitl_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(HitlResult, payload, response)


@router.post("/hitl/{hitl_id}/skip", response_model=HitlResult)
async def skip_hitl(
    hitl_id: UUID,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch,
) -> HitlResult:
    try:
        expected = int(
            if_match.removeprefix('"rv-').removesuffix('"')
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "OPS_ETAG_INVALID",
                "message": "If-Match 格式必须为 \"rv-<version>\"",
            },
        ) from exc
    body = HitlSkipCommand(expected_row_version=expected)
    payload = await _client(request).skip_hitl(
        hitl_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(HitlResult, payload, response)


@router.post("/hitl/{hitl_id}/decision", response_model=HitlResult)
async def decide_diagnostic_query(
    hitl_id: UUID,
    body: DiagnosticQueryApprovalDecision,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> HitlResult:
    payload = await _client(request).decide_diagnostic_query(
        hitl_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(HitlResult, payload, response)


@router.get("/proposals", response_model=ProposalPage)
async def list_proposals(
    request: Request, target_id: UUID | None = None,
    status: str | None = None,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> ProposalPage:
    payload = await _client(request).list_proposals(
        target_id=target_id, status=status, cursor=cursor, limit=limit,
        auth_context=request.state.auth_context,
    )
    return ProposalPage.model_validate(payload)


@router.get("/proposals/{proposal_id}", response_model=ProposalView)
async def get_proposal(
    proposal_id: UUID,
    request: Request,
    response: Response,
) -> ProposalView:
    payload = await _client(request).get_proposal(
        proposal_id, auth_context=request.state.auth_context
    )
    return _validated(ProposalView, payload, response)


@router.post("/proposals/{proposal_id}/reject", response_model=ProposalView)
async def reject_proposal(
    proposal_id: UUID,
    body: RejectionCommand,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> ProposalView:
    payload = await _client(request).reject_proposal(
        proposal_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(ProposalView, payload, response)


@router.post(
    "/proposals/{proposal_id}/approve",
    response_model=ApprovalReceipt,
)
async def approve_proposal(
    proposal_id: UUID,
    body: ApprovalCommand,
    request: Request,
    idempotency_key: IdempotencyKey,
) -> ApprovalReceipt:
    payload = await _client(request).approve_proposal(
        proposal_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return ApprovalReceipt.model_validate(payload)


@router.post(
    "/proposals/{proposal_id}/manual-result",
    response_model=ManualResultReceipt,
)
async def record_manual_result(
    proposal_id: UUID,
    body: ManualResultCommand,
    request: Request,
    idempotency_key: IdempotencyKey,
) -> ManualResultReceipt:
    payload = await _client(request).record_manual_result(
        proposal_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return ManualResultReceipt.model_validate(payload)


@router.post("/runs/{run_id}/cancel", response_model=OpsRunReceipt)
async def cancel_ops_run(
    run_id: UUID,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
    if_match: IfMatch,
) -> OpsRunReceipt:
    try:
        expected = int(
            if_match.removeprefix('"rv-').removesuffix('"')
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "OPS_ETAG_INVALID",
                "message": "If-Match 格式必须为 \"rv-<version>\"",
            },
        ) from exc
    command = OpsCommand(
        command_id=uuid7(),
        idempotency_key=idempotency_key,
        ops_run_id=run_id,
        command=CancelRunCommand(expected_row_version=expected),
    )
    payload = await _client(request).command(
        command, auth_context=request.state.auth_context
    )
    result = OpsRunReceipt(
        **payload,
        events_url=f"{PUBLIC_API_V1}/apps/aiops/runs/{run_id}/events",
    )
    response.headers["ETag"] = f'"rv-{result.row_version}"'
    return result


@router.get("/runs/{run_id}/events")
async def stream_ops_run_events(
    run_id: UUID,
    request: Request,
    last_event_id: str | None = Header(
        default=None, alias="Last-Event-ID"
    ),
) -> StreamingResponse:
    try:
        cursor = int(last_event_id or "0")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "OPS_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 必须是非负整数",
            },
        ) from exc
    if cursor < 0:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "OPS_EVENT_CURSOR_INVALID",
                "message": "Last-Event-ID 不能为负数",
            },
        )
    context = request.state.auth_context
    client = _client(request)

    async def generate():
        nonlocal cursor
        while not await request.is_disconnected():
            page = await client.list_run_events(
                run_id,
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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/targets", response_model=TargetDetail, status_code=201)
async def create_target(
    body: TargetCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    payload = await _client(request).create_target(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.post(
    "/targets/test-connection",
    response_model=TargetConnectionTestResult,
)
async def test_target_connection(
    body: TargetConnectionTest,
    request: Request,
) -> TargetConnectionTestResult:
    payload = await _client(request).test_target_connection(
        body.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return TargetConnectionTestResult.model_validate(payload)


@router.get("/targets", response_model=TargetPage)
async def list_targets(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> TargetPage:
    payload = await _client(request).list_targets(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return TargetPage.model_validate(payload)


@router.get("/targets/{target_id}", response_model=TargetDetail)
async def get_target(
    target_id: UUID, request: Request, response: Response
) -> TargetDetail:
    payload = await _client(request).get_target(
        target_id, auth_context=request.state.auth_context
    )
    return _validated(TargetDetail, payload, response)


@router.patch("/targets/{target_id}", response_model=TargetDetail)
async def patch_target(
    target_id: UUID,
    body: TargetPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> TargetDetail:
    payload = await _client(request).patch_target(
        target_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/diagnostic-credential:rotate", response_model=TargetDetail)
async def rotate_diagnostic_credential(target_id: UUID, body: DatabaseCredentialInput, request: Request, response: Response, if_match: IfMatch, idempotency_key: IdempotencyKey) -> TargetDetail:
    payload = await _client(request).rotate_target_credential(target_id, "diagnostic", body.model_dump(mode="json"), if_match=if_match, idempotency_key=idempotency_key, auth_context=request.state.auth_context)
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/execution-credential:rotate", response_model=TargetDetail)
async def rotate_execution_credential(target_id: UUID, body: DatabaseCredentialInput, request: Request, response: Response, if_match: IfMatch, idempotency_key: IdempotencyKey) -> TargetDetail:
    payload = await _client(request).rotate_target_credential(target_id, "execution", body.model_dump(mode="json"), if_match=if_match, idempotency_key=idempotency_key, auth_context=request.state.auth_context)
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/execution-credential:remove", response_model=TargetDetail)
async def remove_execution_credential(target_id: UUID, request: Request, response: Response, if_match: IfMatch, idempotency_key: IdempotencyKey) -> TargetDetail:
    payload = await _client(request).remove_execution_credential(target_id, if_match=if_match, idempotency_key=idempotency_key, auth_context=request.state.auth_context)
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/diagnostic-credential:remove", response_model=TargetDetail)
async def remove_diagnostic_credential(target_id: UUID, request: Request, response: Response, if_match: IfMatch, idempotency_key: IdempotencyKey) -> TargetDetail:
    payload = await _client(request).remove_diagnostic_credential(target_id, if_match=if_match, idempotency_key=idempotency_key, auth_context=request.state.auth_context)
    return _validated(TargetDetail, payload, response)


@router.delete("/targets/{target_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_target(target_id: UUID, request: Request, if_match: IfMatch, idempotency_key: IdempotencyKey) -> Response:
    await _client(request).delete_target(target_id, if_match=if_match, idempotency_key=idempotency_key, auth_context=request.state.auth_context)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _target_command(
    *,
    target_id: UUID,
    command: str,
    request: Request,
    response: Response,
    if_match: str,
    idempotency_key: str,
) -> TargetDetail:
    payload = await _client(request).command_target(
        target_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.post("/targets/{target_id}/enable", response_model=TargetDetail)
async def enable_target(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    return await _target_command(
        target_id=target_id,
        command="enable",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )

@router.post("/targets/{target_id}/disable", response_model=TargetDetail)
async def disable_target(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    return await _target_command(
        target_id=target_id,
        command="disable",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/targets/{target_id}/connectivity-checks",
    response_model=TargetDetail,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_target_connectivity_check(
    target_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> TargetDetail:
    payload = await _client(request).request_target_connectivity_check(
        target_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(TargetDetail, payload, response)


@router.get(
    "/targets/{target_id}/agent-bindings",
    response_model=tuple[AgentBindingView, ...],
)
async def list_agent_bindings(
    target_id: UUID, request: Request
) -> tuple[AgentBindingView, ...]:
    payload = await _client(request).list_agent_bindings(
        target_id, auth_context=request.state.auth_context
    )
    return tuple(AgentBindingView.model_validate(item) for item in payload)


@router.post(
    "/targets/{target_id}/agent-bindings",
    response_model=AgentBindingView,
    status_code=201,
)
async def create_agent_binding(
    target_id: UUID,
    body: AgentBindingCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> AgentBindingView:
    payload = await _client(request).create_agent_binding(
        target_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.patch(
    "/targets/{target_id}/agent-bindings/{binding_id}",
    response_model=AgentBindingView,
)
async def patch_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    body: AgentBindingPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> AgentBindingView:
    payload = await _client(request).patch_agent_binding(
        target_id,
        binding_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.post(
    "/targets/{target_id}/agent-bindings/{binding_id}/{command}",
    response_model=AgentBindingView,
)
async def command_agent_binding(
    target_id: UUID,
    binding_id: UUID,
    command: Literal["revoke", "restore"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> AgentBindingView:
    payload = await _client(request).command_agent_binding(
        target_id,
        binding_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(AgentBindingView, payload, response)


@router.post(
    "/diagnostic-sources",
    response_model=DiagnosticSourceDetail,
    status_code=201,
)
async def create_diagnostic_source(
    body: DiagnosticSourceCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> DiagnosticSourceDetail:
    payload = await _client(request).create_diagnostic_source(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(DiagnosticSourceDetail, payload, response)


@router.post(
    "/diagnostic-sources/test-connection",
    response_model=DiagnosticSourceConnectionTestResult,
)
async def test_diagnostic_source_connection(
    body: DiagnosticSourceCreate,
    request: Request,
) -> DiagnosticSourceConnectionTestResult:
    payload = await _client(request).test_diagnostic_source_connection(
        body.model_dump(mode="json"),
        auth_context=request.state.auth_context,
    )
    return DiagnosticSourceConnectionTestResult.model_validate(payload)


@router.get("/diagnostic-sources", response_model=DiagnosticSourcePage)
async def list_diagnostic_sources(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> DiagnosticSourcePage:
    payload = await _client(request).list_diagnostic_sources(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return DiagnosticSourcePage.model_validate(payload)


@router.get("/diagnostic-sources/{source_id}", response_model=DiagnosticSourceDetail)
async def get_diagnostic_source(
    source_id: UUID, request: Request, response: Response
) -> DiagnosticSourceDetail:
    payload = await _client(request).get_diagnostic_source(
        source_id, auth_context=request.state.auth_context
    )
    return _validated(DiagnosticSourceDetail, payload, response)


@router.patch(
    "/diagnostic-sources/{source_id}", response_model=DiagnosticSourceDetail
)
async def patch_diagnostic_source(
    source_id: UUID,
    body: DiagnosticSourcePatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> DiagnosticSourceDetail:
    payload = await _client(request).patch_diagnostic_source(
        source_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(DiagnosticSourceDetail, payload, response)


@router.delete(
    "/diagnostic-sources/{source_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_diagnostic_source(
    source_id: UUID,
    request: Request,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> Response:
    await _client(request).delete_diagnostic_source(
        source_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/diagnostic-sources/{source_id}/connectivity-checks",
    response_model=ConnectivityCheckReceipt,
    status_code=status.HTTP_202_ACCEPTED,
)
async def request_diagnostic_source_connectivity_check(
    source_id: UUID,
    request: Request,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> ConnectivityCheckReceipt:
    payload = await _client(request).request_diagnostic_source_connectivity_check(
        source_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return ConnectivityCheckReceipt.model_validate(payload)


@router.post(
    "/diagnostic-sources/{source_id}/webhook-key:rotate",
    response_model=WebhookKeyRotation,
)
async def rotate_diagnostic_source_webhook_key(
    source_id: UUID,
    request: Request,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> WebhookKeyRotation:
    payload = await _client(request).rotate_diagnostic_source_webhook_key(
        source_id,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return WebhookKeyRotation.model_validate(payload)


@router.post(
    "/diagnostic-sources/{source_id}/{command}",
    response_model=DiagnosticSourceDetail,
)
async def command_diagnostic_source(
    source_id: UUID,
    command: Literal["enable", "disable"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> DiagnosticSourceDetail:
    payload = await _client(request).command_diagnostic_source(
        source_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(DiagnosticSourceDetail, payload, response)


@router.get(
    "/targets/{target_id}/source-bindings",
    response_model=tuple[SourceBindingView, ...],
)
async def list_source_bindings(
    target_id: UUID, request: Request
) -> tuple[SourceBindingView, ...]:
    payload = await _client(request).list_source_bindings(
        target_id, auth_context=request.state.auth_context
    )
    return tuple(SourceBindingView.model_validate(item) for item in payload)


@router.post(
    "/targets/{target_id}/source-bindings",
    response_model=SourceBindingView,
    status_code=201,
)
async def create_source_binding(
    target_id: UUID,
    body: SourceBindingCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> SourceBindingView:
    payload = await _client(request).create_source_binding(
        target_id,
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(SourceBindingView, payload, response)


@router.patch(
    "/targets/{target_id}/source-bindings/{binding_id}",
    response_model=SourceBindingView,
)
async def patch_source_binding(
    target_id: UUID,
    binding_id: UUID,
    body: SourceBindingPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> SourceBindingView:
    payload = await _client(request).patch_source_binding(
        target_id,
        binding_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(SourceBindingView, payload, response)


@router.post(
    "/targets/{target_id}/source-bindings/{binding_id}/{command}",
    response_model=SourceBindingView,
)
async def command_source_binding(
    target_id: UUID,
    binding_id: UUID,
    command: Literal["enable", "disable"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> SourceBindingView:
    payload = await _client(request).command_source_binding(
        target_id,
        binding_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(SourceBindingView, payload, response)


@router.post("/policies", response_model=PolicyDetail, status_code=201)
async def create_policy(
    body: PolicyCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> PolicyDetail:
    payload = await _client(request).create_policy(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(PolicyDetail, payload, response)


@router.get("/policies", response_model=PolicyPage)
async def list_policies(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> PolicyPage:
    payload = await _client(request).list_policies(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return PolicyPage.model_validate(payload)


@router.get("/policies/{policy_id}", response_model=PolicyDetail)
async def get_policy(
    policy_id: UUID, request: Request, response: Response
) -> PolicyDetail:
    payload = await _client(request).get_policy(
        policy_id, auth_context=request.state.auth_context
    )
    return _validated(PolicyDetail, payload, response)


@router.post("/policies/{policy_id}/{command}", response_model=PolicyDetail)
async def command_policy(
    policy_id: UUID,
    command: Literal["activate", "retire"],
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> PolicyDetail:
    payload = await _client(request).command_policy(
        policy_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(PolicyDetail, payload, response)


@router.post(
    "/inspection-plans",
    response_model=InspectionPlanDetail,
    status_code=201,
)
async def create_inspection_plan(
    body: InspectionPlanCreate,
    request: Request,
    response: Response,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    payload = await _client(request).create_inspection_plan(
        body.model_dump(mode="json"),
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.get("/inspection-plans", response_model=InspectionPlanPage)
async def list_inspection_plans(
    request: Request,
    resource_status: str | None = Query(default=None, alias="status"),
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> InspectionPlanPage:
    payload = await _client(request).list_inspection_plans(
        status=resource_status,
        cursor=cursor,
        limit=limit,
        auth_context=request.state.auth_context,
    )
    return InspectionPlanPage.model_validate(payload)


@router.get(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def get_inspection_plan(
    plan_id: UUID, request: Request, response: Response
) -> InspectionPlanDetail:
    payload = await _client(request).get_inspection_plan(
        plan_id, auth_context=request.state.auth_context
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.patch(
    "/inspection-plans/{plan_id}", response_model=InspectionPlanDetail
)
async def patch_inspection_plan(
    plan_id: UUID,
    body: InspectionPlanPatch,
    request: Request,
    response: Response,
    if_match: IfMatch,
) -> InspectionPlanDetail:
    payload = await _client(request).patch_inspection_plan(
        plan_id,
        body.model_dump(mode="json", exclude_unset=True),
        if_match=if_match,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


async def _inspection_plan_command(
    *,
    plan_id: UUID,
    command: str,
    request: Request,
    response: Response,
    if_match: str,
    idempotency_key: str,
) -> InspectionPlanDetail:
    payload = await _client(request).command_inspection_plan(
        plan_id,
        command,
        if_match=if_match,
        idempotency_key=idempotency_key,
        auth_context=request.state.auth_context,
    )
    return _validated(InspectionPlanDetail, payload, response)


@router.post(
    "/inspection-plans/{plan_id}/activate",
    response_model=InspectionPlanDetail,
)
async def activate_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="activate",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/inspection-plans/{plan_id}/pause",
    response_model=InspectionPlanDetail,
)
async def pause_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="pause",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )


@router.post(
    "/inspection-plans/{plan_id}/disable",
    response_model=InspectionPlanDetail,
)
async def disable_inspection_plan(
    plan_id: UUID,
    request: Request,
    response: Response,
    if_match: IfMatch,
    idempotency_key: IdempotencyKey,
) -> InspectionPlanDetail:
    return await _inspection_plan_command(
        plan_id=plan_id,
        command="disable",
        request=request,
        response=response,
        if_match=if_match,
        idempotency_key=idempotency_key,
    )
