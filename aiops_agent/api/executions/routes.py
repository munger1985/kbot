"""DB Executor 反向 Claim 内部接口。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from aiops_agent.api.changes.routes import get_service
from aiops_agent.api.dependencies import require_service_scope
from aiops_agent.application.changes import AIOpsChangeService
from platform_core.contracts.aiops.executor import (
    MutationClaimReceipt,
    MutationClaimRequest,
)


router = APIRouter(
    prefix="/internal/v1/aiops/executions",
    tags=["AIOps Executions"],
)
Service = Annotated[AIOpsChangeService, Depends(get_service)]


@router.post(
    "/{execution_id}/claim",
    response_model=MutationClaimReceipt,
)
async def claim_execution(
    execution_id: UUID,
    body: MutationClaimRequest,
    request: Request,
    service: Service,
) -> MutationClaimReceipt:
    require_service_scope(request, "aiops.execution.claim")
    context = request.state.auth_context
    return await service.claim_execution(
        execution_id=execution_id,
        command=body,
        trace_id=context.trace_id,
    )
