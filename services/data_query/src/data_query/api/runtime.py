"""Agent Runtime 委派问数的内部接口。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request, status

from data_query.api.dependencies import (
    actor_id_from_context,
    domain_id_from_context,
    get_auth_context,
    require_scope,
)
from data_query.application import DataQueryRuntimeService
from data_query.contracts import (
    CreateDataQueryRun,
    DataQueryResultView,
    DataQueryPlanningContext,
    DataQueryRunReceipt,
    DataQueryRunView,
)
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/data-query/runs", tags=["Data Query Runtime"])


def get_service(request: Request) -> DataQueryRuntimeService:
    return request.app.state.runtime_service


Service = Annotated[DataQueryRuntimeService, Depends(get_service)]
Auth = Annotated[AuthContext, Depends(get_auth_context)]


@router.post("", response_model=DataQueryRunReceipt, status_code=status.HTTP_202_ACCEPTED)
async def create_run(body: CreateDataQueryRun, request: Request, service: Service, context: Auth) -> DataQueryRunReceipt:
    require_scope(request, "data_query.delegate")
    return await service.create_run(
        domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context),
        actor_roles=context.roles, trace_id=context.trace_id, command=body,
    )


@router.get("/planning-context/{agent_id}", response_model=DataQueryPlanningContext)
async def get_planning_context(
    agent_id: UUID, consumer_app_id: str, agent_version_id: UUID,
    request: Request, service: Service, context: Auth,
) -> DataQueryPlanningContext:
    require_scope(request, "data_query.delegate")
    return await service.get_planning_context(
        domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context),
        actor_roles=context.roles, consumer_app_id=consumer_app_id,
        agent_id=agent_id, agent_version_id=agent_version_id,
    )


@router.get("/{data_query_run_id}", response_model=DataQueryRunView)
async def get_run(data_query_run_id: UUID, request: Request, service: Service, context: Auth) -> DataQueryRunView:
    require_scope(request, "data_query.delegate")
    return await service.get_run(data_query_run_id=data_query_run_id, domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context))


@router.get("/{data_query_run_id}/result", response_model=DataQueryResultView)
async def get_result(data_query_run_id: UUID, request: Request, service: Service, context: Auth) -> DataQueryResultView:
    require_scope(request, "data_query.delegate")
    return await service.get_result(data_query_run_id=data_query_run_id, domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context))


@router.post("/{data_query_run_id}/cancel", response_model=DataQueryRunView, status_code=status.HTTP_202_ACCEPTED)
async def cancel_run(data_query_run_id: UUID, request: Request, service: Service, context: Auth) -> DataQueryRunView:
    require_scope(request, "data_query.delegate")
    return await service.cancel_run(data_query_run_id=data_query_run_id, domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context))
