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

APP_MEMBER_MANAGE_PERMISSIONS = {
    "knowledge_retrieval": "knowledge_retrieval:member_manage",
    "km_asset": "km_asset:member_manage",
    "aiops": "aiops:member_manage",
}


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
    max_security_level: int = Field(default=1, ge=0, le=3)

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        return _validate_password(value)


class UserUpdatePayload(_Payload):
    display_name: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] | None = None
    max_security_level: int | None = Field(default=None, ge=0, le=3)


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


def _domain_actor(request: Request) -> tuple[int, str]:
    context = get_auth_context(request)
    if not context.domain_id or not context.asserted_user_id:
        raise HTTPException(
            401,
            {
                "code": "USER_CONTEXT_REQUIRED",
                "message": "用户管理需要 Domain 和用户上下文",
            },
        )
    return int(context.domain_id), context.asserted_user_id


async def _has_permission(
    request: Request,
    *,
    app_id: str,
    domain_id: int,
    actor_id: str,
    permission_code: str,
) -> bool:
    access = cast(
        AccessControlService, request.app.state.access_control_service
    )
    try:
        await access.require(
            app_id=app_id,
            domain_id=domain_id,
            user_id=actor_id,
            permission_code=permission_code,
        )
    except AccessDeniedError:
        return False
    return True


async def _require(request: Request, permission_code: str) -> tuple[int, str]:
    domain_id, actor_id = _domain_actor(request)
    if not await _has_permission(
        request,
        app_id="platform",
        domain_id=domain_id,
        actor_id=actor_id,
        permission_code=permission_code,
    ):
        raise HTTPException(
            403,
            {
                "code": "PLATFORM_PERMISSION_DENIED",
                "permission": permission_code,
            },
        )
    return domain_id, actor_id


async def _require_user_creator(request: Request) -> tuple[int, str, bool]:
    """允许平台管理员或当前 Domain 的任一应用成员管理员建号。"""
    domain_id, actor_id = _domain_actor(request)
    if await _has_permission(
        request,
        app_id="platform",
        domain_id=domain_id,
        actor_id=actor_id,
        permission_code="platform:user_manage",
    ):
        return domain_id, actor_id, True
    for app_id, permission_code in APP_MEMBER_MANAGE_PERMISSIONS.items():
        if await _has_permission(
            request,
            app_id=app_id,
            domain_id=domain_id,
            actor_id=actor_id,
            permission_code=permission_code,
        ):
            return domain_id, actor_id, False
    raise HTTPException(
        403,
        {
            "code": "USER_CREATION_PERMISSION_DENIED",
            "message": "当前用户没有创建应用成员账号的权限",
        },
    )


async def _require_membership_manager(
    request: Request,
    *,
    app_id: str,
    target_domain_id: int,
) -> tuple[int, str]:
    """允许平台管理员，或限定在本 App、本 Domain 的成员管理员。"""
    domain_id, actor_id = _domain_actor(request)
    if await _has_permission(
        request,
        app_id="platform",
        domain_id=domain_id,
        actor_id=actor_id,
        permission_code="platform:user_manage",
    ):
        return domain_id, actor_id

    permission_code = APP_MEMBER_MANAGE_PERMISSIONS.get(app_id)
    if permission_code is None or target_domain_id != domain_id:
        raise HTTPException(
            403,
            {
                "code": "APP_MEMBERSHIP_SCOPE_DENIED",
                "message": "应用管理员只能管理当前 Domain 下本应用的成员",
            },
        )
    if not await _has_permission(
        request,
        app_id=app_id,
        domain_id=domain_id,
        actor_id=actor_id,
        permission_code=permission_code,
    ):
        raise HTTPException(
            403,
            {
                "code": "APP_PERMISSION_DENIED",
                "permission": permission_code,
            },
        )
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
    _, _, is_platform_manager = await _require_user_creator(request)
    if not is_platform_manager and payload.max_security_level != 1:
        raise HTTPException(
            403,
            {
                "code": "USER_SECURITY_LEVEL_DENIED",
                "message": "应用成员管理员只能创建默认安全等级用户",
            },
        )
    try:
        return await _management(request).create_user(
            user_id=payload.user_id.strip(),
            display_name=payload.display_name,
            password=payload.password,
            status=payload.status,
            must_change_password=payload.must_change_password,
            max_security_level=payload.max_security_level,
        )
    except AccessManagementError as exc:
        raise HTTPException(
            exc.status_code, {"code": exc.code, "message": str(exc)}
        ) from exc


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
            max_security_level=payload.max_security_level,
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
        _, actor_id = await _require_membership_manager(
            request,
            app_id=app_id,
            target_domain_id=payload.domain_id,
        )
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
