"""Proposal 查询、驳回和 Advisory 人工结果 Internal API。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.changes import AIOpsChangeService
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ManualResultCommand,
    ManualResultReceipt,
    ProposalView,
    RejectionCommand,
)


router = APIRouter(
    prefix="/internal/v1/aiops/proposals",
    tags=["AIOps Changes"],
)


def get_service(request: Request) -> AIOpsChangeService:
    return request.app.state.change_service


Service = Annotated[AIOpsChangeService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _scope(request: Request, context: AuthContext) -> tuple[int, int]:
    if context.domain_id is None:
        raise RuntimeError("AIOps 请求缺少 Domain")
    return (
        request.app.state.runtime.settings.platform.app_id,
        int(context.domain_id),
    )


@router.get("/{proposal_id}", response_model=ProposalView)
async def get_proposal(
    proposal_id: UUID,
    request: Request,
    service: Service,
    context: Auth,
) -> ProposalView:
    require_service_scope(request, "aiops.approve")
    app_id, domain_id = _scope(request, context)
    return await service.get_proposal(
        proposal_id=proposal_id,
        app_id=app_id,
        domain_id=domain_id,
    )


@router.post("/{proposal_id}/reject", response_model=ProposalView)
async def reject_proposal(
    proposal_id: UUID,
    body: RejectionCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> ProposalView:
    require_service_scope(request, "aiops.approve")
    app_id, domain_id = _scope(request, context)
    return await service.reject_proposal(
        proposal_id=proposal_id,
        app_id=app_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        command=body,
        trace_id=context.trace_id,
    )


@router.post(
    "/{proposal_id}/manual-result",
    response_model=ManualResultReceipt,
)
async def record_manual_result(
    proposal_id: UUID,
    body: ManualResultCommand,
    request: Request,
    service: Service,
    context: Auth,
) -> ManualResultReceipt:
    require_service_scope(request, "aiops.approve")
    app_id, domain_id = _scope(request, context)
    return await service.record_manual_result(
        proposal_id=proposal_id,
        app_id=app_id,
        domain_id=domain_id,
        actor_id=context.asserted_user_id or context.client_id,
        command=body,
        idempotency_key=request.headers.get(
            "Idempotency-Key", str(body.expected_row_version)
        ),
        trace_id=context.trace_id,
    )
