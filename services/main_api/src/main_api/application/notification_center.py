"""无用户目录依赖的通知中心用例。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from uuid import UUID

from platform_core.notifications import EVENT_TYPES, event_definition
from main_api.application.access_control import AccessControlService
from platform_core.identity import uuid7
from main_api.entities.access_control import AccessAuditEventEntity
from main_api.application.aiops_notification_routes import AIOPS_OUTPUT_EVENTS, AIOPS_RESOURCE_EVENTS, aiops_notification_route


class NotificationCenterError(ValueError):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def serialize_entity(entity: Any) -> dict[str, Any]:
    return {
        column.name: (
            value.isoformat() if isinstance(value, datetime)
            else str(value) if isinstance(value, UUID)
            else int(value) if hasattr(value, "as_integer_ratio") and not isinstance(value, (int, float, bool))
            else value
        )
        for column in entity.__table__.columns
        if (value := getattr(entity, column.name)) is not None
    }


class NotificationCenterService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def personal(self, method, *, actor_id, **params):
        """个人收件箱跨当前获授权空间聚合，不以登录时选中的空间分仓。"""
        async with self._uow_factory() as uow:
            domains = tuple([domain for domain in await uow.access.list_active_domain_ids(actor_id)
                if await uow.domains.exists_active(domain_id=domain)])
        access = AccessControlService(uow_factory=self._uow_factory)
        event_types = {}
        for domain in domains:
            permissions = set()
            for app_id in {permission.split(":")[0] for definition in EVENT_TYPES.values()
                           for permission in definition.read_permissions}:
                permissions.update((await access.snapshot(app_id=app_id, domain_id=domain, user_id=actor_id)).permissions)
            event_types[domain] = tuple(code for code, definition in EVENT_TYPES.items()
                                       if set(definition.read_permissions) <= permissions)
        if method in {"get_notification", "set_read"}:
            async with self._uow_factory() as uow:
                domain = await self._repo(uow).inbox_domain_for_actor(
                    inbox_id=params["inbox_id"], actor_id=actor_id)
            if domain not in domains:
                raise NotificationCenterError("NOTIFICATION_NOT_FOUND", "通知不存在", 404)
            return await getattr(self, method)(domain_id=domain, actor_id=actor_id,
                allowed_event_types=event_types[domain], **params)
        results = [await getattr(self, method)(domain_id=domain, actor_id=actor_id,
            **({"allowed_event_types": event_types[domain]} if method in {
                "summary", "list_notifications", "stream_events", "mark_many_read"} else {}), **params) for domain in domains]
        if method == "summary":
            return {key: sum(row[key] for row in results)
                    for key in ("unread", "open_work_items", "active_operations")}
        if method == "mark_many_read":
            return {"updated": sum(row["updated"] for row in results)}
        rows = [row for result in results for row in result]
        if method in {"list_notifications", "stream_events"}:
            return sorted(rows, key=lambda row: int(row["event_sequence"]),
                reverse=method == "list_notifications")[:params.get("limit", 100)]
        return sorted(rows, key=lambda row: row.get("updated_at", ""), reverse=True)[:params.get("limit", 100)]

    @staticmethod
    async def _serialize_inbox(repository, rows, *, domain_id, actor_id):
        """在授权页内批量补白名单定位，不泄露整个事件信封。"""
        ids = [row.inbox_id for row in rows if row.event_type in AIOPS_OUTPUT_EVENTS | AIOPS_RESOURCE_EVENTS.keys()]
        payloads = await repository.inbox_route_payloads(
            domain_id=domain_id, actor_id=actor_id, inbox_ids=ids) if ids else {}
        return [serialize_entity(row) | aiops_notification_route(
            row, payloads.get(row.inbox_id), domain_id=domain_id) for row in rows]

    async def summary(self, *, domain_id: int, actor_id: str, allowed_event_types=None) -> dict[str, int]:
        async with self._uow_factory() as uow:
            return await self._repo(uow).summary(allowed_event_types=allowed_event_types,
                domain_id=domain_id, actor_id=actor_id,
            )

    async def list_notifications(
        self, *, domain_id: int, actor_id: str,
        limit: int, before_sequence: int | None, allowed_event_types=None,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).list_inbox(allowed_event_types=allowed_event_types,
                domain_id=domain_id, actor_id=actor_id,
                limit=limit, before_sequence=before_sequence,
            )
            return await self._serialize_inbox(self._repo(uow), rows, domain_id=domain_id, actor_id=actor_id)

    async def resource_notifications(
        self, *, domain_id: int, actor_id: str, resource_type: str,
        resource_id: str, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """返回运行组合视图中属于当前 Actor 的资源通知。"""
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).list_resource_inbox(
                domain_id=domain_id, actor_id=actor_id,
                resource_type=resource_type, resource_id=resource_id,
                limit=limit,
            )
            return await self._serialize_inbox(self._repo(uow), rows, domain_id=domain_id, actor_id=actor_id)

    async def stream_events(
        self, *, domain_id: int, actor_id: str,
        after_sequence: int, limit: int = 100, allowed_event_types=None,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).stream_inbox(allowed_event_types=allowed_event_types,
                domain_id=domain_id, actor_id=actor_id,
                after_sequence=after_sequence, limit=limit,
            )
            return await self._serialize_inbox(self._repo(uow), rows, domain_id=domain_id, actor_id=actor_id)

    async def get_notification(
        self, *, inbox_id: UUID, domain_id: int, actor_id: str, allowed_event_types=None,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await self._repo(uow).get_inbox(allowed_event_types=allowed_event_types,
                inbox_id=inbox_id, domain_id=domain_id, actor_id=actor_id,
            )
            if row is None:
                raise NotificationCenterError(
                    "NOTIFICATION_NOT_FOUND", "通知不存在", 404,
                )
            return (await self._serialize_inbox(self._repo(uow), [row], domain_id=domain_id, actor_id=actor_id))[0]

    async def set_read(
        self, *, inbox_id: UUID, domain_id: int, actor_id: str,
        read: bool, expected_row_version: int, allowed_event_types=None,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await self._repo(uow).get_inbox(allowed_event_types=allowed_event_types,
                inbox_id=inbox_id, domain_id=domain_id,
                actor_id=actor_id, lock=True,
            )
            if row is None:
                raise NotificationCenterError(
                    "NOTIFICATION_NOT_FOUND", "通知不存在", 404,
                )
            if int(row.row_version) != expected_row_version:
                raise NotificationCenterError(
                    "NOTIFICATION_VERSION_CONFLICT", "通知状态已变化", 409,
                )
            row.read_at = datetime.now(timezone.utc) if read else None
            row.row_version = int(row.row_version) + 1
            await uow.commit()
            return (await self._serialize_inbox(self._repo(uow), [row], domain_id=domain_id, actor_id=actor_id))[0]

    async def mark_many_read(
        self, *, inbox_ids: list[UUID], domain_id: int, actor_id: str, allowed_event_types=None,
    ) -> dict[str, int]:
        async with self._uow_factory() as uow:
            updated = await self._repo(uow).mark_many_read(allowed_event_types=allowed_event_types,
                inbox_ids=inbox_ids, domain_id=domain_id, actor_id=actor_id,
            )
            await uow.commit()
            return {"updated": updated}

    async def list_work_items(
        self, *, domain_id: int, actor_id: str, status: str, limit: int,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).list_work_items(
                domain_id=domain_id, actor_id=actor_id,
                status=status, limit=limit,
            )
            return [serialize_entity(row) for row in rows]

    async def list_operations(
        self, *, domain_id: int, actor_id: str, limit: int,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).list_operations(
                domain_id=domain_id, actor_id=actor_id, limit=limit,
            )
            return [serialize_entity(row) for row in rows]

    async def watch_operation(
        self, *, operation_id: UUID, domain_id: int,
        actor_id: str, notify_terminal: bool,
    ) -> None:
        async with self._uow_factory() as uow:
            repository = self._repo(uow)
            operation = await repository.get_operation(
                operation_id=operation_id, domain_id=domain_id,
            )
            if operation is None:
                raise NotificationCenterError(
                    "OPERATION_NOT_FOUND", "后台任务不存在", 404,
                )
            if operation.initiator_actor_id != actor_id:
                # 关注关系不能成为获得其他用户任务可见性的入口。
                raise NotificationCenterError("OPERATION_NOT_FOUND", "后台任务不存在", 404)
            await repository.watch(
                operation_id=operation_id, domain_id=domain_id,
                actor_id=actor_id, notify_terminal=notify_terminal,
            )
            await uow.commit()

    async def unwatch_operation(
        self, *, operation_id: UUID, domain_id: int, actor_id: str,
    ) -> None:
        async with self._uow_factory() as uow:
            deleted = await self._repo(uow).unwatch(
                operation_id=operation_id, domain_id=domain_id, actor_id=actor_id,
            )
            if not deleted:
                raise NotificationCenterError(
                    "OPERATION_WATCH_NOT_FOUND", "未关注该后台任务", 404,
                )
            await uow.commit()

    async def preferences(
        self, *, domain_id: int, actor_id: str,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            return [
                serialize_entity(row)
                for row in await self._repo(uow).preferences(
                    domain_id=domain_id, actor_id=actor_id,
                )
            ]

    async def set_preference(
        self, *, domain_id: int, actor_id: str,
        event_type: str, enabled: bool,
    ) -> None:
        event_definition(event_type)
        async with self._uow_factory() as uow:
            await self._repo(uow).set_preference(
                domain_id=domain_id, actor_id=actor_id,
                event_type=event_type, enabled=enabled,
            )
            await uow.commit()

    async def forget_actor(self, *, domain_id: int, actor_id: str) -> dict[str, int]:
        async with self._uow_factory() as uow:
            result = await self._repo(uow).forget_actor(
                domain_id=domain_id, actor_id=actor_id,
            )
            await uow.commit()
            return result

    async def quarantine(self, *, domain_id: int, limit: int) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).quarantine(domain_id=domain_id, limit=limit)
            allowed = {"outbox_id", "producer_service", "event_type", "domain_id", "status",
                "attempt_count", "last_error_code", "created_at", "updated_at"}
            return [{key: value for key, value in serialize_entity(row).items() if key in allowed}
                for row in rows]

    async def retry_quarantined(self, *, domain_id: int, outbox_id: UUID, actor_id: str) -> None:
        async with self._uow_factory() as uow:
            changed = await self._repo(uow).retry_quarantined(
                domain_id=domain_id, outbox_id=outbox_id,
            )
            if not changed:
                raise NotificationCenterError(
                    "NOTIFICATION_QUARANTINE_NOT_FOUND", "隔离事件不存在", 404,
                )
            await uow.access.add_access_audit_event(AccessAuditEventEntity(
                event_id=str(uuid7()), actor_id=actor_id, app_id="platform",
                action="NOTIFICATION_RETRY", resource_type="notification_outbox",
                resource_id=str(outbox_id), summary_json=json.dumps({"domain_id": domain_id}),
                affected_member_count=0))
            await uow.commit()

    @staticmethod
    def _repo(uow):
        if uow.notifications is None:
            raise RuntimeError("Notification Repository 未初始化")
        return uow.notifications
