"""通知 Worker 专用的工作流状态读取接口。"""

from fastapi import APIRouter, Depends, HTTPException, Request

from aiops_agent.api.dependencies import get_aiops_auth_context, require_service_scope
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.contracts.aiops.notification_projection import (
    WorkflowProjectionRequest, WorkflowProjectionSnapshot, NotificationResourceRequest, NotificationResourceSnapshot,
)

router = APIRouter(prefix="/internal/v1/aiops/notifications", tags=["AIOps Notifications"])

@router.post(":resource", response_model=NotificationResourceSnapshot)
async def notification_resource(body: NotificationResourceRequest, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context)):
    require_service_scope(request, "aiops.notification.reconcile")
    if context.principal_kind != PrincipalKind.SERVICE:
        raise HTTPException(403, {"code": "AIOPS_NOTIFICATION_SERVICE_REQUIRED"})
    return await request.app.state.workflow_projection_service.resolve_notification_resource(body)


@router.post(":projection", response_model=WorkflowProjectionSnapshot)
async def workflow_projection(body: WorkflowProjectionRequest, request: Request,
    context: AuthContext = Depends(get_aiops_auth_context)):
    require_service_scope(request, "aiops.notification.reconcile")
    if context.principal_kind != PrincipalKind.SERVICE:
        raise HTTPException(403, {"code": "AIOPS_NOTIFICATION_SERVICE_REQUIRED"})
    return await request.app.state.workflow_projection_service.read(body)
