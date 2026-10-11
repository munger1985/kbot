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
from aiops_agent.application.responsibility_groups import ResponsibilityGroupService
from aiops_agent.application.work_items import WorkItemService
from platform_core.contracts import AuthContext
from platform_core.contracts.aiops import (
    ResponsibilityGroupCreate,
    ResponsibilityGroupMemberUpsert,
    ResponsibilityGroupPage,
    ResponsibilityGroupPatch,
    ResponsibilityGroupView,
    WorkItemAssignment,
    WorkItemCompletion,
    WorkItemCreate,
    WorkItemPage,
    WorkItemRouteRun,
    WorkItemSummary,
    WorkItemTransition,
    WorkItemVerification,
    WorkItemVersionCommand,
    WorkItemView,
)


router = APIRouter(prefix="/internal/v1/aiops/work-items", tags=["AIOps Work Items"])
group_router = APIRouter(prefix="/internal/v1/aiops/responsibility-groups", tags=["AIOps Responsibility Groups"])
Auth = Annotated[AuthContext, Depends(get_aiops_auth_context)]


def _service(request: Request) -> WorkItemService:
    return request.app.state.work_item_service


def _group_service(request: Request) -> ResponsibilityGroupService:
    return request.app.state.responsibility_group_service


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


@router.post("/{work_item_id}:claim", response_model=WorkItemView)
async def claim_work_item(work_item_id: UUID, body: WorkItemVersionCommand, request: Request, context: Auth) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).claim(work_item_id=work_item_id, body=body, scope=_scope(context))


@router.post("/{work_item_id}:return-to-group", response_model=WorkItemView)
async def return_work_item(work_item_id: UUID, body: WorkItemVersionCommand, request: Request, context: Auth) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).return_to_group(work_item_id=work_item_id, body=body, scope=_scope(context))


@router.post("/{work_item_id}:complete", response_model=WorkItemView)
async def complete_work_item(work_item_id: UUID, body: WorkItemCompletion, request: Request, context: Auth) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).complete(work_item_id=work_item_id, body=body, scope=_scope(context))


@router.post("/{work_item_id}:verify", response_model=WorkItemView)
async def verify_work_item(work_item_id: UUID, body: WorkItemVerification, request: Request, context: Auth) -> WorkItemView:
    require_service_scope(request, "aiops.manage")
    return await _service(request).verify(work_item_id=work_item_id, body=body, scope=_scope(context))


@group_router.get("", response_model=ResponsibilityGroupPage)
async def list_groups(request: Request, context: Auth) -> ResponsibilityGroupPage:
    require_service_scope(request, "aiops.run")
    return await _group_service(request).list_groups(scope=_scope(context))


@group_router.post("", response_model=ResponsibilityGroupView, status_code=201)
async def create_group(body: ResponsibilityGroupCreate, request: Request, context: Auth) -> ResponsibilityGroupView:
    require_service_scope(request, "aiops.manage")
    return await _group_service(request).create_group(body=body, scope=_scope(context))


@group_router.get("/{group_id}", response_model=ResponsibilityGroupView)
async def get_group(group_id: UUID, request: Request, context: Auth) -> ResponsibilityGroupView:
    require_service_scope(request, "aiops.run")
    return await _group_service(request).get_group(group_id=group_id, scope=_scope(context))


@group_router.patch("/{group_id}", response_model=ResponsibilityGroupView)
async def patch_group(group_id: UUID, body: ResponsibilityGroupPatch, request: Request, context: Auth) -> ResponsibilityGroupView:
    require_service_scope(request, "aiops.manage")
    return await _group_service(request).patch_group(group_id=group_id, body=body, scope=_scope(context))


@group_router.put("/{group_id}/members/{user_id}", response_model=ResponsibilityGroupView)
async def put_group_member(group_id: UUID, user_id: str, body: ResponsibilityGroupMemberUpsert, request: Request, context: Auth) -> ResponsibilityGroupView:
    require_service_scope(request, "aiops.manage")
    return await _group_service(request).put_member(group_id=group_id, user_id=user_id, body=body, scope=_scope(context))


@group_router.delete("/{group_id}/members/{user_id}", response_model=ResponsibilityGroupView)
async def remove_group_member(group_id: UUID, user_id: str, request: Request, context: Auth) -> ResponsibilityGroupView:
    require_service_scope(request, "aiops.manage")
    return await _group_service(request).remove_member(group_id=group_id, user_id=user_id, scope=_scope(context))
