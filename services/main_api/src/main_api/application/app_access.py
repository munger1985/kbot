"""Main API 对所有业务 App 的统一授权入口。"""

from __future__ import annotations

from fastapi import HTTPException, Request

from platform_core.authorization import (
    get_app_authorization_policy,
    resolve_business_app_id,
)
from platform_core.contracts import PrincipalKind
from platform_core.security import get_auth_context

from .access_control import AccessControlService, AccessSnapshot
from .app_api_key import AppApiKeyError


def _resolve_app_actor(request: Request, *, app_id: str) -> tuple[int, str]:
    """从受信身份中解析固定 App、Domain 和 Actor。"""

    get_app_authorization_policy(app_id)
    context = get_auth_context(request)
    if context.principal_kind not in {
        PrincipalKind.PORTAL,
        PrincipalKind.APP_API_CLIENT,
    }:
        raise HTTPException(403, {"code": "APP_PRINCIPAL_UNSUPPORTED"})
    if context.app_id and context.app_id != app_id:
        raise HTTPException(403, {"code": "APP_CONTEXT_MISMATCH"})
    try:
        domain_id = int(context.domain_id or "")
    except ValueError as exc:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"}) from exc
    actor_id = context.asserted_user_id or context.client_id
    if domain_id < 1 or not actor_id:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    return domain_id, actor_id


async def _app_access_snapshot(
    request: Request, *, app_id: str
) -> tuple[int, str, AccessSnapshot]:
    """读取当前 App/Domain 的唯一权限快照。"""

    domain_id, actor_id = _resolve_app_actor(request, app_id=app_id)
    cache = getattr(request.state, "app_access_snapshots", None)
    if cache is None:
        cache = {}
        request.state.app_access_snapshots = cache
    cache_key = (app_id, domain_id, actor_id)
    cached = cache.get(cache_key)
    if cached is not None:
        return domain_id, actor_id, cached
    service: AccessControlService = request.app.state.access_control_service
    snapshot = await service.snapshot(
        app_id=app_id,
        domain_id=domain_id,
        user_id=actor_id,
    )
    cache[cache_key] = snapshot
    return domain_id, actor_id, snapshot


def _permission_denied(permission: str) -> HTTPException:
    return HTTPException(
        403,
        {"code": "APP_PERMISSION_DENIED", "permission": permission},
    )


def _require_machine_route_scope(request: Request, *, app_id: str) -> None:
    """机器请求仅按核心路由注册表校验 Scope。"""

    context = get_auth_context(request)
    if context.principal_kind != PrincipalKind.APP_API_CLIENT:
        return
    policy = get_app_authorization_policy(app_id)
    prefix = f"/api/v1/apps/{policy.public_slug}/"
    path = request.url.path
    if not path.startswith(prefix):
        raise AppApiKeyError(
            "APP_API_KEY_CONTEXT_MISMATCH",
            "API Client 只能访问绑定 App 的公开 Main API",
        )
    required_scope = policy.required_scope(
        method=request.method,
        relative_path=path.removeprefix(prefix).strip("/"),
    )
    if required_scope is None or required_scope not in context.scopes:
        raise AppApiKeyError(
            "APP_API_KEY_SCOPE_DENIED",
            "API Client Scope 不允许访问该公开业务接口",
        )


async def authorize_app_request(
    request: Request, *, app_id: str, permission: str
) -> tuple[int, str, AccessSnapshot]:
    """以统一规则校验人类权限和机器 Scope。"""

    policy = get_app_authorization_policy(app_id)
    if not permission.startswith(f"{policy.app_id}:"):
        raise RuntimeError(f"权限不属于 App {app_id}：{permission}")
    _require_machine_route_scope(request, app_id=app_id)
    domain_id, actor_id, snapshot = await _app_access_snapshot(
        request, app_id=app_id
    )
    if policy.use_permission not in snapshot.permissions:
        raise _permission_denied(policy.use_permission)
    if permission not in snapshot.permissions:
        raise _permission_denied(permission)
    return domain_id, actor_id, snapshot


async def authorize_app_request_any(
    request: Request, *, app_id: str, permissions: tuple[str, ...]
) -> tuple[int, str, AccessSnapshot]:
    """要求当前主体至少拥有一项同 App 权限。"""

    if not permissions:
        raise RuntimeError("至少需要声明一项 App 权限")
    policy = get_app_authorization_policy(app_id)
    if any(
        not permission.startswith(f"{policy.app_id}:")
        for permission in permissions
    ):
        raise RuntimeError(f"权限不属于 App {app_id}")
    _require_machine_route_scope(request, app_id=app_id)
    domain_id, actor_id, snapshot = await _app_access_snapshot(
        request, app_id=app_id
    )
    matched = tuple(
        permission
        for permission in permissions
        if permission in snapshot.permissions
    )
    if policy.use_permission not in snapshot.permissions:
        raise _permission_denied(policy.use_permission)
    if not matched:
        raise _permission_denied(permissions[0])
    return domain_id, actor_id, snapshot


async def authorize_business_app_route(request: Request) -> None:
    """对全部公开业务 App 路由强制执行注册、Domain 和 use 权限。"""

    prefix = "/api/v1/apps/"
    if not request.url.path.startswith(prefix):
        return
    relative = request.url.path.removeprefix(prefix).strip("/")
    parts = relative.split("/") if relative else []
    if len(parts) >= 3 and parts[1] == "auth" and parts[2] in {
        "login",
        "password",
    }:
        return
    if not parts or not parts[0]:
        raise HTTPException(404, {"code": "APP_NOT_FOUND"})
    try:
        app_id = resolve_business_app_id(parts[0])
    except ValueError as exc:
        raise HTTPException(404, {"code": "APP_NOT_FOUND"}) from exc
    policy = get_app_authorization_policy(app_id)
    await authorize_app_request(
        request,
        app_id=app_id,
        permission=policy.use_permission,
    )
