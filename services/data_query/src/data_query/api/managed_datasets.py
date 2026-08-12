"""只允许业务 App 调和代码拥有的数据集。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from data_query.api.dependencies import actor_id_from_context, domain_id_from_context, get_auth_context, require_scope
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/data-query/managed-datasets", tags=["Data Query Managed Datasets"])
Auth = Annotated[AuthContext, Depends(get_auth_context)]


@router.post("/km-asset/reconcile")
async def reconcile_km_asset(request: Request, context: Auth):
    require_scope(request, "data_query.managed")
    return await request.app.state.managed_dataset_service.reconcile_km_asset(domain_id=domain_id_from_context(context), actor_id=actor_id_from_context(context))
