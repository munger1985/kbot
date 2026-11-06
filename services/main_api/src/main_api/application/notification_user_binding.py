"""用户管理事务中的可选个人绑定；绑定不授予业务权限。"""

from uuid import UUID, uuid4
from datetime import datetime

from main_api.application.notification_delivery import ExternalDeliveryError, now_utc, public_record


def binding_record(row):
    return {key: str(value) if isinstance(value, UUID) else value.isoformat() if isinstance(value, datetime) else value
            for key, value in public_record("binding", row).items()}


async def personal_binding(repo, *, scope, user_id):
    rows = await repo.my_bindings(scope=scope, actor_id=str(user_id))
    active = [row for row in rows if row["target_type"] == "open_id" and row["status"] != "DISABLED"]
    if len(active) > 1:
        raise ExternalDeliveryError("NOTIFICATION_BINDING_CONFLICT")
    return binding_record(active[0]) if active else None


async def set_personal_binding(repo, *, scope, user_id, actor_id, open_id,
                               expected_binding_id=None, expected_version=None, creating=False):
    """父连接行锁串行化身份占用；替换保留历史，旧任务失效。"""
    open_id = open_id.strip() if open_id else None
    if open_id and (not open_id.startswith("ou_") or not 4 <= len(open_id) <= 256):
        raise ExternalDeliveryError("NOTIFICATION_EXTERNAL_ID_INVALID", 422)
    # 新建普通账号不依赖渠道开关、凭据或外部表。
    if creating and not open_id:
        return None
    connection = await repo.connection_for_provider(scope=scope, lock=True)
    if not connection:
        if open_id:
            raise ExternalDeliveryError("NOTIFICATION_CONNECTION_NOT_CONFIGURED", 422)
        if expected_binding_id is not None:
            raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
        return None
    rows = await repo.bindings_for_connection(scope=scope, connection_id=connection["id"])
    active = [row for row in rows if row["actor_id"] == str(user_id) and row["status"] != "DISABLED"]
    if len(active) > 1 or (creating and active):
        raise ExternalDeliveryError("NOTIFICATION_BINDING_CONFLICT")
    current = active[0] if active else None
    if not creating and (str(current["id"]) if current else None, current["row_version"] if current else None) != (
        str(expected_binding_id) if expected_binding_id else None, expected_version):
        raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
    if current and current["external_id"] == open_id:
        return binding_record(current)
    if open_id and any(row["status"] != "DISABLED" and row["external_id"] == open_id
                       and row["actor_id"] != str(user_id) for row in rows):
        raise ExternalDeliveryError("NOTIFICATION_BINDING_CONFLICT")
    now = now_utc()
    if current:
        if not await repo.change("binding", scope=scope, object_id=current["id"],
            expected_version=current["row_version"], values={"status": "DISABLED",
                "challenge_hash": None, "challenge_ref": None}, actor_id=str(actor_id), now=now):
            raise ExternalDeliveryError("NOTIFICATION_VERSION_CONFLICT")
        await repo.audit(scope=scope, action="binding.disable", object_id=current["id"],
            actor_id=str(actor_id), outcome="USER_EDIT", now=now)
    if not open_id:
        return None
    row = await repo.add("binding", scope=scope, actor_id=str(actor_id), now=now, values={
        "id": uuid4(), "connection_id": connection["id"], "actor_id": str(user_id),
        "target_type": "open_id", "external_id": open_id, "status": "UNVERIFIED",
        "verified_at": None, "challenge_attempts": 0})
    await repo.audit(scope=scope, action="binding.create", object_id=row["id"],
        actor_id=str(actor_id), outcome="UNVERIFIED", now=now)
    return binding_record(row)
