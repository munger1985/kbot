"""平台用户、角色和成员授权管理公开接口。"""

from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.application import (
    AccessControlService,
    AccessDeniedError,
    AccessManagementError,
    AccessManagementService,
)
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(prefix=f"{PUBLIC_API_V1}/admin", tags=["Access Management"])


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _validate_password(value: str) -> str:
    if not (
        any(char.islower() for char in value)
        and any(char.isupper() for char in value)
        and any(char.isdigit() for char in value)
        and any(not char.isalnum() for char in value)
    ):
        raise ValueError("密码必须同时包含大小写字母、数字和特殊字符")
    return value


class UserCreatePayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=256)
    password: str = Field(min_length=12, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"
    must_change_password: bool = True

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return _validate_password(value)


class UserUpdatePayload(_Payload):
    display_name: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] | None = None


class PasswordResetPayload(_Payload):
    password: str = Field(min_length=12, max_length=256)
    must_change_password: bool = True

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return _validate_password(value)


class MembershipPayload(_Payload):
    domain_id: int = Field(gt=0)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"


class RoleCreatePayload(_Payload):
    app_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    role_code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    display_name: str = Field(min_length=1, max_length=256)
    permissions: tuple[str, ...] = Field(default=(), max_length=200)


class RoleUpdatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    status: Literal["ACTIVE", "DISABLED"]
    permissions: tuple[str, ...] = Field(default=(), max_length=200)


def _management(request: Request) -> AccessManagementService:
    return cast(
        AccessManagementService, request.app.state.access_management_service
    )


async def _require(request: Request, permission_code: str) -> tuple[int, str]:
    context = get_auth_context(request)
    if not context.domain_id or not context.asserted_user_id:
        raise HTTPException(
            401,
            {
                "code": "USER_CONTEXT_REQUIRED",
                "message": "用户管理需要 Domain 和用户上下文",
            },
        )
    domain_id = int(context.domain_id)
    actor_id = context.asserted_user_id
    access = cast(
        AccessControlService, request.app.state.access_control_service
    )
    try:
        await access.require(
            app_id="platform",
            domain_id=domain_id,
            user_id=actor_id,
            permission_code=permission_code,
        )
    except AccessDeniedError as exc:
        raise HTTPException(
            403,
            {
                "code": "PLATFORM_PERMISSION_DENIED",
                "permission": permission_code,
            },
        ) from exc
    return domain_id, actor_id


async def _call(request: Request, permission: str, operation):
    await _require(request, permission)
    try:
        return await operation()
    except AccessManagementError as exc:
        raise HTTPException(
            exc.status_code, {"code": exc.code, "message": str(exc)}
        ) from exc


@router.get("/users")
async def list_users(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    search: str | None = Query(default=None, max_length=256),
    user_status: Literal["ACTIVE", "DISABLED"] | None = Query(
        default=None, alias="status"
    ),
):
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).list_users(
            offset=offset,
            limit=limit,
            search=search,
            status=user_status,
        ),
    )


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(payload: UserCreatePayload, request: Request):
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).create_user(
            user_id=payload.user_id.strip(),
            display_name=payload.display_name,
            password=payload.password,
            status=payload.status,
            must_change_password=payload.must_change_password,
        ),
    )


@router.get("/users/{user_id}")
async def get_user(user_id: str, request: Request):
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).get_user(user_id=user_id),
    )


@router.patch("/users/{user_id}")
async def update_user(
    user_id: str, payload: UserUpdatePayload, request: Request
):
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).update_user(
            user_id=user_id,
            display_name=payload.display_name,
            display_name_provided="display_name" in payload.model_fields_set,
            status=payload.status,
        ),
    )


@router.post("/users/{user_id}/password")
async def reset_password(
    user_id: str, payload: PasswordResetPayload, request: Request
):
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).reset_password(
            user_id=user_id,
            password=payload.password,
            must_change_password=payload.must_change_password,
        ),
    )


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, request: Request):
    """物理删除普通用户、登录凭据及全部成员授权。"""
    return await _call(
        request,
        "platform:user_manage",
        lambda: _management(request).delete_user(user_id=user_id),
    )


@router.put("/users/{user_id}/memberships/{app_id}/{role_code}")
async def set_membership(
    user_id: str,
    app_id: str,
    role_code: str,
    payload: MembershipPayload,
    request: Request,
):
    async def operation():
        _, actor_id = await _require(request, "platform:user_manage")
        return await _management(request).set_membership(
            app_id=app_id,
            domain_id=payload.domain_id,
            user_id=user_id,
            role_code=role_code,
            status=payload.status,
            actor_id=actor_id,
        )

    try:
        return await operation()
    except AccessManagementError as exc:
        raise HTTPException(
            exc.status_code, {"code": exc.code, "message": str(exc)}
        ) from exc


@router.get("/permissions")
async def list_permissions(request: Request, app_id: str | None = None):
    return await _call(
        request,
        "platform:role_manage",
        lambda: _management(request).list_permissions(app_id=app_id),
    )


@router.get("/roles")
async def list_roles(request: Request, app_id: str | None = None):
    return await _call(
        request,
        "platform:role_manage",
        lambda: _management(request).list_roles(app_id=app_id),
    )


@router.post("/roles", status_code=status.HTTP_201_CREATED)
async def create_role(payload: RoleCreatePayload, request: Request):
    return await _call(
        request,
        "platform:role_manage",
        lambda: _management(request).create_role(
            app_id=payload.app_id,
            role_code=payload.role_code,
            display_name=payload.display_name,
            permission_codes=payload.permissions,
        ),
    )


@router.put("/roles/{app_id}/{role_code}")
async def update_role(
    app_id: str, role_code: str, payload: RoleUpdatePayload, request: Request
):
    return await _call(
        request,
        "platform:role_manage",
        lambda: _management(request).update_role(
            app_id=app_id,
            role_code=role_code,
            display_name=payload.display_name,
            status=payload.status,
            permission_codes=payload.permissions,
        ),
    )


@router.delete("/roles/{app_id}/{role_code}")
async def delete_role(app_id: str, role_code: str, request: Request):
    """逻辑删除自定义角色；平台保留角色不允许删除。"""
    return await _call(
        request,
        "platform:role_manage",
        lambda: _management(request).delete_role(
            app_id=app_id, role_code=role_code
        ),
    )


__all__ = ["router"]
