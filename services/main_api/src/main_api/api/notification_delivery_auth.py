"""KBot 外部通知权限；禁止服务主体冒充人类管理员。"""

from fastapi import HTTPException
from main_api.application.access_control import AccessDeniedError
from main_api.application import authorize_app_request
from platform_core.contracts import PrincipalKind
from platform_core.notifications.delivery import DeliveryScope


async def external_entry_actor(request):
    context = request.state.auth_context
    if context.principal_kind != PrincipalKind.PORTAL or not context.asserted_user_id:
        raise HTTPException(403, {"code": "NOTIFICATION_HUMAN_CONTEXT_REQUIRED"})
    return str(context.asserted_user_id)


async def external_capabilities(request):
    actor_id = await external_entry_actor(request)
    snapshot = await request.app.state.access_control_service.platform_snapshot(user_id=actor_id)
    return {"capabilities": [key for key, code in {
        "channel": "platform:notification_channel_manage", "binding": "platform:notification_binding_manage",
        "read": "platform:notification_delivery_read", "retry": "platform:notification_delivery_retry",
    }.items() if code in snapshot.permissions]}


async def external_context(request, *, permission, domain_id):
    context = request.state.auth_context
    actor_id = str(context.asserted_user_id or "").strip()
    if context.principal_kind != PrincipalKind.PORTAL or not actor_id:
        raise HTTPException(403, {"code": "NOTIFICATION_HUMAN_CONTEXT_REQUIRED"})
    if str(context.domain_id) != domain_id or not domain_id.isdigit() or int(domain_id) < 1:
        raise HTTPException(403, {"code": "NOTIFICATION_DOMAIN_MISMATCH"})
    codes = {"channel": "platform:notification_channel_manage", "binding": "platform:notification_binding_manage",
             "read": "platform:notification_delivery_read", "retry": "platform:notification_delivery_retry"}
    try:
        if permission in codes:
            await request.app.state.access_control_service.require_platform(user_id=actor_id, permission_code=codes[permission])
        else:
            await authorize_app_request(request, app_id="aiops", permission="aiops:use")
    except AccessDeniedError as exc:
        raise HTTPException(403, {"code": "NOTIFICATION_ACCESS_DENIED"}) from exc
    return DeliveryScope(domain_id=domain_id), actor_id
