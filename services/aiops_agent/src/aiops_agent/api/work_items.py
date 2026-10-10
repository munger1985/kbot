"""DBA 工作项 Internal API。"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from aiops_agent.api.dependencies import (
    get_aiops_auth_context,
    require_service_scope,
)
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.work_items import WorkItemService
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    WorkItemAssignment,
    WorkItemCreate,
    WorkItemPage,
    WorkItemRouteRun,
    WorkItemSummary,
    WorkItemTransition,
    WorkItemView,
)


router = APIRouter(prefix="/internal/v1/aiops/work-items", tags=["AIOps Work Items"])
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _service(request: Request) -> WorkItemService:
    return request.app.state.work_item_service


def _scope(context: AuthContext) -> ConfigurationScope:
    return ConfigurationScope.from_auth(auth_context=context)


@router.get("", response_model=WorkItemPage)
async def list_work_items(
    request: Request,
    context: Auth,
    status: str | None = None,
    priority: str | None = None,
    target_id: UUID | None = None,
    assignee_user_id: str | None = None,
    unassigned: bool = False,
    overdue: bool = False,
    cursor: str | None = Query(default=None, max_length=2048),
    limit: int = Query(default=50, ge=1, le=200),
) -> WorkItemPage:
    require_service_scope(request, "aiops.run")
    return await _service(request).list_items(
        scope=_scope(context), status=status, priority=priority,
        target_id=target_id, assignee_user_id=assignee_user_id,
        unassigned=unassigned, cursor=cursor, limit=limit,
        overdue=overdue,
    )


@router.post("", response_model=WorkItemView, status_code=201)
async def create_work_item(
    body: WorkItemCreate, request: Request, context: Auth,
) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).create_item(body=body, scope=_scope(context))


@router.post(":route-run", response_model=tuple[WorkItemSummary, ...])
async def route_run_to_work_items(
    body: WorkItemRouteRun, request: Request, context: Auth,
) -> tuple[WorkItemSummary, ...]:
    require_service_scope(request, "aiops.manage")
    return await _service(request).route_run(body=body, scope=_scope(context))


@router.get("/{work_item_id}", response_model=WorkItemView)
async def get_work_item(
    work_item_id: UUID, request: Request, context: Auth,
) -> WorkItemView:
    require_service_scope(request, "aiops.run")
    return await _service(request).get_item(
        work_item_id=work_item_id, scope=_scope(context)
    )


@router.patch("/{work_item_id}/assignment", response_model=WorkItemView)
async def assign_work_item(
    work_item_id: UUID, body: WorkItemAssignment,
    request: Request, context: Auth,
) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).assign(
        work_item_id=work_item_id, body=body, scope=_scope(context)
    )


@router.post("/{work_item_id}/transitions", response_model=WorkItemView)
async def transition_work_item(
    work_item_id: UUID, body: WorkItemTransition,
    request: Request, context: Auth,
) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).transition(
        work_item_id=work_item_id, body=body, scope=_scope(context)
    )
