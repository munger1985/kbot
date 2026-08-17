"""平台治理与 App 内部成员、角色管理公开接口。"""

from typing import Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.application import AccessControlService, AccessDeniedError, AccessManagementError, AccessManagementService
from platform_core.contracts import PUBLIC_API_V1
from platform_core.security import get_auth_context


router = APIRouter(tags=["Access Management"])
APP_SLUG_TO_ID = {
    "knowledge-retrieval": "knowledge_retrieval",
    "km-asset": "km_asset",
}


def _canonical_app_id(value: str) -> str:
    return APP_SLUG_TO_ID.get(value, value)


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


class PlatformUserCreatePayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=256)
    password: str = Field(min_length=12, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] = "ACTIVE"
    must_change_password: bool = True
    max_security_level: int = Field(default=1, ge=0, le=3)
    platform_role_codes: tuple[str, ...] = ()

    _password = field_validator("password")(_validate_password)


class UserUpdatePayload(_Payload):
    display_name: str | None = Field(default=None, max_length=256)
    status: Literal["ACTIVE", "DISABLED"] | None = None
    max_security_level: int | None = Field(default=None, ge=0, le=3)


class PasswordResetPayload(_Payload):
    password: str = Field(min_length=12, max_length=256)
    must_change_password: bool = True

    _password = field_validator("password")(_validate_password)


class RoleBindingPayload(_Payload):
    role_code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    scope_mode: Literal["ALL_APP_DOMAINS", "SELECTED_DOMAINS"] = "SELECTED_DOMAINS"
    domain_ids: tuple[int, ...] = Field(default=(), max_length=200)

    @field_validator("domain_ids")
    @classmethod
    def normalize_domain_ids(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(domain_id <= 0 for domain_id in value):
            raise ValueError("Domain ID 必须为正整数")
        return tuple(dict.fromkeys(value))


class AppUserCreatePayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=256)
    password: str = Field(min_length=12, max_length=256)
    must_change_password: bool = True
    max_security_level: int = Field(default=1, ge=0, le=3)
    role_bindings: tuple[RoleBindingPayload, ...] = Field(min_length=1, max_length=50)

    _password = field_validator("password")(_validate_password)


class InitialAppAdminPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    display_name: str | None = Field(default=None, max_length=256)
    password: str = Field(min_length=12, max_length=256)
    must_change_password: bool = False
    max_security_level: int = Field(default=3, ge=0, le=3)

    _password = field_validator("password")(_validate_password)


class PlatformAppGrantPayload(_Payload):
    role_bindings: tuple[RoleBindingPayload, ...] = Field(min_length=1, max_length=50)


class PlatformUserRolesPayload(_Payload):
    role_codes: tuple[str, ...] = Field(max_length=50)


class ApplicationStatusPayload(_Payload):
    status: Literal["ACTIVE", "DISABLED"]


class RoleCreatePayload(_Payload):
    role_code: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    display_name: str = Field(min_length=1, max_length=256)
    permissions: tuple[str, ...] = Field(default=(), max_length=200)


class RoleUpdatePayload(_Payload):
    display_name: str = Field(min_length=1, max_length=256)
    status: Literal["ACTIVE", "DISABLED"]
    permissions: tuple[str, ...] = Field(default=(), max_length=200)


def _management(request: Request) -> AccessManagementService:
    return cast(AccessManagementService, request.app.state.access_management_service)


def _access(request: Request) -> AccessControlService:
    return cast(AccessControlService, request.app.state.access_control_service)


def _actor(request: Request) -> str:
    actor_id = get_auth_context(request).asserted_user_id
    if not actor_id:
        raise HTTPException(401, {"code": "USER_CONTEXT_REQUIRED", "message": "缺少用户上下文"})
    return actor_id


async def _require_platform(request: Request, permission_code: str) -> str:
    actor_id = _actor(request)
    try:
        await _access(request).require_platform(user_id=actor_id, permission_code=permission_code)
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "PLATFORM_PERMISSION_DENIED", "permission": permission_code}) from exc
    return actor_id


async def _require_app(request: Request, *, app_id: str, permission_code: str):
    context = get_auth_context(request)
    actor_id = context.asserted_user_id
    if not actor_id or not context.domain_id:
        raise HTTPException(401, {"code": "APP_CONTEXT_REQUIRED", "message": "App 管理需要用户和 Domain 上下文"})
    domain_id = int(context.domain_id)
    if context.app_id and context.app_id != app_id:
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH", "message": "登录 Token 绑定的 App 与请求不一致"})
    try:
        snapshot = await _access(request).require(
            app_id=app_id, domain_id=domain_id, user_id=actor_id,
            permission_code=permission_code,
        )
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "APP_PERMISSION_DENIED", "permission": permission_code}) from exc
    return actor_id, domain_id, snapshot


async def _call(operation):
    try:
        return await operation()
    except AccessManagementError as exc:
        raise HTTPException(exc.status_code, {"code": exc.code, "message": str(exc)}) from exc


def _bindings(payload: tuple[RoleBindingPayload, ...]) -> tuple[dict[str, object], ...]:
    return tuple(item.model_dump() for item in payload)


@router.get(f"{PUBLIC_API_V1}/platform/users")
async def list_platform_users(
    request: Request, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
    search: str | None = Query(None, max_length=256),
    user_status: Literal["ACTIVE", "DISABLED"] | None = Query(None, alias="status"),
):
    await _require_platform(request, "platform:user_manage")
    return await _call(lambda: _management(request).list_users(
        offset=offset, limit=limit, search=search, status=user_status, account_origin="PLATFORM"
    ))


@router.post(f"{PUBLIC_API_V1}/platform/users", status_code=status.HTTP_201_CREATED)
async def create_platform_user(payload: PlatformUserCreatePayload, request: Request):
    actor_id = await _require_platform(request, "platform:user_manage")
    actor_security_level = await _access(request).user_max_security_level(user_id=actor_id)
    actor_permissions = (await _access(request).platform_snapshot(user_id=actor_id)).permissions
    return await _call(lambda: _management(request).create_platform_user(
        user_id=payload.user_id.strip(), display_name=payload.display_name,
        password=payload.password, status=payload.status,
        must_change_password=payload.must_change_password,
        max_security_level=payload.max_security_level,
        platform_role_codes=payload.platform_role_codes, actor_id=actor_id,
        actor_security_level=actor_security_level,
        actor_permissions=actor_permissions,
    ))


@router.get(f"{PUBLIC_API_V1}/platform/users/{{user_id}}")
async def get_platform_user(user_id: str, request: Request):
    await _require_platform(request, "platform:user_manage")
    result = await _call(lambda: _management(request).get_user(user_id=user_id))
    if result["account_origin"] != "PLATFORM":
        raise HTTPException(404, {"code": "PLATFORM_USER_NOT_FOUND"})
    return result


@router.patch(f"{PUBLIC_API_V1}/platform/users/{{user_id}}")
async def update_platform_user(user_id: str, payload: UserUpdatePayload, request: Request):
    actor_id = await _require_platform(request, "platform:user_manage")
    actor_security_level = await _access(request).user_max_security_level(user_id=actor_id)
    return await _call(lambda: _management(request).update_user(
        user_id=user_id, display_name=payload.display_name,
        display_name_provided="display_name" in payload.model_fields_set,
        status=payload.status, max_security_level=payload.max_security_level,
        expected_origin="PLATFORM",
        actor_security_level=actor_security_level,
    ))


@router.post(f"{PUBLIC_API_V1}/platform/users/{{user_id}}/password")
async def reset_platform_user_password(user_id: str, payload: PasswordResetPayload, request: Request):
    await _require_platform(request, "platform:user_manage")
    user = await _call(lambda: _management(request).get_user(user_id=user_id))
    if user["account_origin"] != "PLATFORM":
        raise HTTPException(403, {"code": "PLATFORM_USER_REQUIRED"})
    return await _call(lambda: _management(request).reset_password(
        user_id=user_id, password=payload.password, must_change_password=payload.must_change_password
    ))


@router.delete(f"{PUBLIC_API_V1}/platform/users/{{user_id}}")
async def delete_platform_user(user_id: str, request: Request):
    await _require_platform(request, "platform:user_manage")
    return await _call(lambda: _management(request).delete_user(user_id=user_id, expected_origin="PLATFORM"))


@router.put(f"{PUBLIC_API_V1}/platform/users/{{user_id}}/roles")
async def set_platform_user_roles(user_id: str, payload: PlatformUserRolesPayload, request: Request):
    actor_id = await _require_platform(request, "platform:user_manage")
    snapshot = await _access(request).platform_snapshot(user_id=actor_id)
    return await _call(lambda: _management(request).set_platform_user_roles(
        user_id=user_id, role_codes=payload.role_codes, actor_id=actor_id,
        assignable_permissions=snapshot.permissions,
    ))


@router.get(f"{PUBLIC_API_V1}/platform/permissions")
async def list_platform_permissions(request: Request):
    await _require_platform(request, "platform:role_manage")
    return await _call(lambda: _management(request).list_permissions(app_id="platform"))


@router.get(f"{PUBLIC_API_V1}/platform/roles")
async def list_platform_roles(request: Request):
    await _require_platform(request, "platform:role_manage")
    return await _call(lambda: _management(request).list_roles(app_id="platform"))


@router.post(f"{PUBLIC_API_V1}/platform/roles", status_code=status.HTTP_201_CREATED)
async def create_platform_role(payload: RoleCreatePayload, request: Request):
    actor_id = await _require_platform(request, "platform:role_manage")
    snapshot = await _access(request).platform_snapshot(user_id=actor_id)
    return await _call(lambda: _management(request).create_role(
        app_id="platform", role_code=payload.role_code,
        display_name=payload.display_name, permission_codes=payload.permissions,
        assignable_permissions=snapshot.permissions,
    ))


@router.put(f"{PUBLIC_API_V1}/platform/roles/{{role_code}}")
async def update_platform_role(role_code: str, payload: RoleUpdatePayload, request: Request):
    actor_id = await _require_platform(request, "platform:role_manage")
    snapshot = await _access(request).platform_snapshot(user_id=actor_id)
    return await _call(lambda: _management(request).update_role(
        app_id="platform", role_code=role_code,
        display_name=payload.display_name, status=payload.status,
        permission_codes=payload.permissions,
        assignable_permissions=snapshot.permissions,
    ))


@router.delete(f"{PUBLIC_API_V1}/platform/roles/{{role_code}}")
async def delete_platform_role(role_code: str, request: Request):
    await _require_platform(request, "platform:role_manage")
    return await _call(lambda: _management(request).delete_role(app_id="platform", role_code=role_code))


@router.get(f"{PUBLIC_API_V1}/platform/apps")
async def list_applications(request: Request):
    await _require_platform(request, "platform:app_manage")
    return await _call(lambda: _management(request).list_applications())


@router.patch(f"{PUBLIC_API_V1}/platform/apps/{{app_id}}/status")
async def set_application_status(app_id: str, payload: ApplicationStatusPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_platform(request, "platform:app_manage")
    return await _call(lambda: _management(request).set_application_status(app_id=app_id, status=payload.status))


@router.put(f"{PUBLIC_API_V1}/platform/apps/{{app_id}}/domains/{{domain_id}}")
async def assign_app_domain(app_id: str, domain_id: int, request: Request):
    app_id = _canonical_app_id(app_id)
    actor_id = await _require_platform(request, "platform:app_manage")
    return await _call(lambda: _management(request).assign_app_domain(app_id=app_id, domain_id=domain_id, actor_id=actor_id))


@router.post(f"{PUBLIC_API_V1}/platform/apps/{{app_id}}/initial-admin", status_code=status.HTTP_201_CREATED)
async def create_initial_app_admin(app_id: str, payload: InitialAppAdminPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    actor_id = await _require_platform(request, "platform:app_manage")
    actor_security_level = await _access(request).user_max_security_level(user_id=actor_id)
    return await _call(lambda: _management(request).create_initial_app_admin(
        app_id=app_id, user_id=payload.user_id.strip(), display_name=payload.display_name,
        password=payload.password, must_change_password=payload.must_change_password,
        max_security_level=payload.max_security_level, actor_id=actor_id,
        actor_security_level=actor_security_level,
    ))


@router.post(f"{PUBLIC_API_V1}/platform/apps/{{app_id}}/initial-admin/password")
async def reset_initial_app_admin_password(app_id: str, payload: PasswordResetPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_platform(request, "platform:app_manage")
    return await _call(lambda: _management(request).reset_initial_app_admin_password(
        app_id=app_id, password=payload.password, must_change_password=payload.must_change_password
    ))


@router.put(f"{PUBLIC_API_V1}/platform/users/{{user_id}}/app-grants/{{app_id}}")
async def set_platform_app_grant(user_id: str, app_id: str, payload: PlatformAppGrantPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    actor_id = await _require_platform(request, "platform:app_grant_manage")
    return await _call(lambda: _management(request).set_platform_app_grant(
        user_id=user_id, app_id=app_id, role_bindings=_bindings(payload.role_bindings), actor_id=actor_id
    ))


@router.delete(f"{PUBLIC_API_V1}/platform/users/{{user_id}}/app-grants/{{app_id}}")
async def revoke_platform_app_grant(user_id: str, app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_platform(request, "platform:app_grant_manage")
    return await _call(lambda: _management(request).revoke_platform_app_grant(user_id=user_id, app_id=app_id))


@router.get(f"{PUBLIC_API_V1}/apps/{{app_id}}/members")
async def list_app_members(app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:member_manage")
    return await _call(lambda: _management(request).list_app_users(app_id=app_id))


@router.post(f"{PUBLIC_API_V1}/apps/{{app_id}}/members", status_code=status.HTTP_201_CREATED)
async def create_app_member(app_id: str, payload: AppUserCreatePayload, request: Request):
    app_id = _canonical_app_id(app_id)
    actor_id, _, snapshot = await _require_app(request, app_id=app_id, permission_code=f"{app_id}:member_manage")
    actor_security_level = await _access(request).user_max_security_level(user_id=actor_id)
    return await _call(lambda: _management(request).create_app_user(
        app_id=app_id, user_id=payload.user_id.strip(), display_name=payload.display_name,
        password=payload.password, must_change_password=payload.must_change_password,
        max_security_level=payload.max_security_level,
        role_bindings=_bindings(payload.role_bindings), actor_id=actor_id,
        actor_security_level=actor_security_level,
        actor_permissions=snapshot.permissions,
    ))


@router.patch(f"{PUBLIC_API_V1}/apps/{{app_id}}/members/{{user_id}}")
async def update_app_member(app_id: str, user_id: str, payload: UserUpdatePayload, request: Request):
    app_id = _canonical_app_id(app_id)
    actor_id, _, _ = await _require_app(request, app_id=app_id, permission_code=f"{app_id}:member_manage")
    actor_security_level = await _access(request).user_max_security_level(user_id=actor_id)
    return await _call(lambda: _management(request).update_user(
        user_id=user_id, display_name=payload.display_name,
        display_name_provided="display_name" in payload.model_fields_set,
        status=payload.status, max_security_level=payload.max_security_level,
        expected_origin="APP", expected_app_id=app_id,
        actor_security_level=actor_security_level,
    ))


@router.post(f"{PUBLIC_API_V1}/apps/{{app_id}}/members/{{user_id}}/password")
async def reset_app_member_password(app_id: str, user_id: str, payload: PasswordResetPayload, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:member_manage")
    return await _call(lambda: _management(request).reset_app_user_password(
        app_id=app_id, user_id=user_id, password=payload.password,
        must_change_password=payload.must_change_password,
    ))


@router.put(f"{PUBLIC_API_V1}/apps/{{app_id}}/members/{{user_id}}/role-bindings")
async def set_app_member_role_bindings(
    app_id: str, user_id: str, payload: PlatformAppGrantPayload, request: Request
):
    app_id = _canonical_app_id(app_id)
    actor_id, _, snapshot = await _require_app(
        request, app_id=app_id, permission_code=f"{app_id}:member_manage"
    )
    return await _call(lambda: _management(request).set_app_user_role_bindings(
        app_id=app_id, user_id=user_id,
        role_bindings=_bindings(payload.role_bindings), actor_id=actor_id,
        actor_permissions=snapshot.permissions,
    ))


@router.delete(f"{PUBLIC_API_V1}/apps/{{app_id}}/members/{{user_id}}")
async def delete_app_member(app_id: str, user_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:member_manage")
    return await _call(lambda: _management(request).delete_user(
        user_id=user_id, expected_origin="APP", expected_app_id=app_id
    ))


@router.get(f"{PUBLIC_API_V1}/apps/{{app_id}}/permissions")
async def list_app_permissions(app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:role_manage")
    return await _call(lambda: _management(request).list_permissions(app_id=app_id))


@router.get(f"{PUBLIC_API_V1}/apps/{{app_id}}/roles")
async def list_app_roles(app_id: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:role_manage")
    return await _call(lambda: _management(request).list_roles(app_id=app_id))


@router.post(f"{PUBLIC_API_V1}/apps/{{app_id}}/roles", status_code=status.HTTP_201_CREATED)
async def create_app_role(app_id: str, payload: RoleCreatePayload, request: Request):
    app_id = _canonical_app_id(app_id)
    _, _, snapshot = await _require_app(request, app_id=app_id, permission_code=f"{app_id}:role_manage")
    return await _call(lambda: _management(request).create_role(
        app_id=app_id, role_code=payload.role_code, display_name=payload.display_name,
        permission_codes=payload.permissions,
        assignable_permissions=snapshot.permissions,
    ))


@router.put(f"{PUBLIC_API_V1}/apps/{{app_id}}/roles/{{role_code}}")
async def update_app_role(app_id: str, role_code: str, payload: RoleUpdatePayload, request: Request):
    app_id = _canonical_app_id(app_id)
    _, _, snapshot = await _require_app(request, app_id=app_id, permission_code=f"{app_id}:role_manage")
    return await _call(lambda: _management(request).update_role(
        app_id=app_id, role_code=role_code, display_name=payload.display_name,
        status=payload.status, permission_codes=payload.permissions,
        assignable_permissions=snapshot.permissions,
    ))


@router.delete(f"{PUBLIC_API_V1}/apps/{{app_id}}/roles/{{role_code}}")
async def delete_app_role(app_id: str, role_code: str, request: Request):
    app_id = _canonical_app_id(app_id)
    await _require_app(request, app_id=app_id, permission_code=f"{app_id}:role_manage")
    return await _call(lambda: _management(request).delete_role(app_id=app_id, role_code=role_code))


__all__ = ["router"]
