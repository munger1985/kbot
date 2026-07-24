"""平台级 Domain 管理公开契约。"""

from typing import cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from main_api.application import DomainConflictError, DomainManagementService
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/domains", tags=["Domains"])


class DomainCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)


def _service(request: Request) -> DomainManagementService:
    service = getattr(request.app.state, "domain_management_service", None)
    if service is None:
        raise RuntimeError("Domain Management Service 尚未初始化")
    return cast(DomainManagementService, service)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_domain(payload: DomainCreateRequest, request: Request):
    context = get_auth_context(request)
    try:
        return await _service(request).create(
            name=payload.name,
            description=payload.description,
            actor_id=context.asserted_user_id or context.client_id,
        )
    except DomainConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "DOMAIN_NAME_CONFLICT",
                "message": str(exc),
            },
        ) from exc
