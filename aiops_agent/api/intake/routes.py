"""经 Main API 转发的原始监控事件入口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.application.monitoring import MonitorWebhookIntakeService
from platform_core.contracts.aiops import (
    MonitorWebhookEnvelope,
    MonitorWebhookReceipt,
)


router = APIRouter(
    prefix="/internal/v1/aiops/intake", tags=["AIOps Monitor Intake"]
)


def get_service(request: Request) -> MonitorWebhookIntakeService:
    return request.app.state.monitor_intake_service


Service = Annotated[MonitorWebhookIntakeService, Depends(get_service)]


@router.post(
    "/monitor-events",
    response_model=MonitorWebhookReceipt,
    status_code=202,
)
async def intake_monitor_event(
    body: MonitorWebhookEnvelope,
    request: Request,
    service: Service,
) -> MonitorWebhookReceipt:
    require_service_scope(request, "aiops.monitor.intake")
    return await service.intake(body)
