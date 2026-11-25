"""KBot 通知治理归 AIOps 应用管理员；个人入口不授予治理权限。"""

import re

from fastapi import HTTPException
from main_api.application.app_access import authorize_app_request
from platform_core.contracts import PrincipalKind
from platform_core.notifications.delivery import DeliveryScope


_UUID_SEGMENT = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
_SELF_VERIFY_PATH = re.compile(
    rf"/api/v1/notification-delivery/(?:me/bindings/{_UUID_SEGMENT}/challenge|bindings/{_UUID_SEGMENT}/confirm)"
)


def accepts_workspace_notification_session(method: str, path: str) -> bool:
    """App 来源用户也能在根会话中操作本人绑定，不开放渠道治理。"""
    return (method, path) in {
        ("GET", "/api/v1/notification-delivery/me/bindings"),
        ("PUT", "/api/v1/notification-delivery/me/bindings/feishu"),
    } or (method == "POST" and _SELF_VERIFY_PATH.fullmatch(path) is not None)


async def external_entry_actor(request):
    context = request.state.auth_context
    if context.principal_kind != PrincipalKind.PORTAL or not context.asserted_user_id:
        raise HTTPException(403, {"code": "NOTIFICATION_HUMAN_CONTEXT_REQUIRED"})
    return str(context.asserted_user_id)


async def external_capabilities(request):
    await external_entry_actor(request)
    _, _, snapshot = await authorize_app_request(
        request, app_id="aiops", permission="aiops:use"
    )
    # 角色取当前有效成员快照，不信任 Token 中可能已撤销的角色或平台管理员身份。
    return {"capabilities": ["channel", "binding", "read", "retry"]
            if "app_admin" in snapshot.roles else []}


async def external_context(request, *, permission):
    actor_id = await external_entry_actor(request)
    if permission == "self":
        claims = getattr(request.state, "user_token_claims", None)
        if claims is None or claims.user_id != actor_id or claims.must_change_password:
            raise HTTPException(403, {"code": "NOTIFICATION_USER_SESSION_REQUIRED"})
        allowed = await request.app.state.external_notification_service.authority.actor_active(
            scope=DeliveryScope(), actor_id=actor_id)
    else:
        allowed = permission in (await external_capabilities(request))["capabilities"]
    if not allowed:
        raise HTTPException(403, {"code": "NOTIFICATION_ACCESS_DENIED"})
    return DeliveryScope(), actor_id
