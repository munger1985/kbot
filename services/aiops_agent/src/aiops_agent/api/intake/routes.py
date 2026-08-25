"""经 Main API 转发的原始信号事件入口。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.application.diagnostic_sources import SignalEventIntakeService
from platform_core.contracts.aiops import (
    SignalEventEnvelope,
    SignalEventIntakeReceipt,
)


router = APIRouter(
    prefix="/internal/v1/aiops/intake", tags=["AIOps Signal Intake"]
)


def get_service(request: Request) -> SignalEventIntakeService:
    return request.app.state.signal_intake_service


Service = Annotated[SignalEventIntakeService, Depends(get_service)]


@router.post(
    "/signal-events",
    response_model=SignalEventIntakeReceipt,
    status_code=202,
)
async def intake_signal_event(
    body: SignalEventEnvelope,
    request: Request,
    service: Service,
) -> SignalEventIntakeReceipt:
    require_service_scope(request, "aiops.monitor.intake")
    return await service.intake(body)
