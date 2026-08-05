"""无用户目录依赖的通知中心用例。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from platform_core.notifications import event_definition


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

    async def summary(self, *, domain_id: int, actor_id: str) -> dict[str, int]:
        async with self._uow_factory() as uow:
            return await self._repo(uow).summary(
                domain_id=domain_id, actor_id=actor_id,
            )

    async def list_notifications(
        self, *, domain_id: int, actor_id: str,
        limit: int, before_sequence: int | None,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).list_inbox(
                domain_id=domain_id, actor_id=actor_id,
                limit=limit, before_sequence=before_sequence,
            )
            return [serialize_entity(row) for row in rows]

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
            return [serialize_entity(row) for row in rows]

    async def stream_events(
        self, *, domain_id: int, actor_id: str,
        after_sequence: int, limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            rows = await self._repo(uow).stream_inbox(
                domain_id=domain_id, actor_id=actor_id,
                after_sequence=after_sequence, limit=limit,
            )
            return [serialize_entity(row) for row in rows]

    async def get_notification(
        self, *, inbox_id: UUID, domain_id: int, actor_id: str,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await self._repo(uow).get_inbox(
                inbox_id=inbox_id, domain_id=domain_id, actor_id=actor_id,
            )
            if row is None:
                raise NotificationCenterError(
                    "NOTIFICATION_NOT_FOUND", "通知不存在", 404,
                )
            return serialize_entity(row)

    async def set_read(
        self, *, inbox_id: UUID, domain_id: int, actor_id: str,
        read: bool, expected_row_version: int,
    ) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            row = await self._repo(uow).get_inbox(
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
            return serialize_entity(row)

    async def mark_many_read(
        self, *, inbox_ids: list[UUID], domain_id: int, actor_id: str,
    ) -> dict[str, int]:
        async with self._uow_factory() as uow:
            updated = await self._repo(uow).mark_many_read(
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
            return [serialize_entity(row) for row in rows]

    async def retry_quarantined(self, *, domain_id: int, outbox_id: UUID) -> None:
        async with self._uow_factory() as uow:
            changed = await self._repo(uow).retry_quarantined(
                domain_id=domain_id, outbox_id=outbox_id,
            )
            if not changed:
                raise NotificationCenterError(
                    "NOTIFICATION_QUARANTINE_NOT_FOUND", "隔离事件不存在", 404,
                )
            await uow.commit()

    @staticmethod
    def _repo(uow):
        if uow.notifications is None:
            raise RuntimeError("Notification Repository 未初始化")
        return uow.notifications
