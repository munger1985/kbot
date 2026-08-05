"""将共享 Outbox 事件幂等投影为 Inbox、待办和后台任务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger

from main_api.entities import (
    BackgroundOperationEntity,
    NotificationInboxEntity,
    WorkItemEntity,
)
from platform_core.notifications import NotificationEnvelope, event_definition


_TERMINAL = {"SUCCEEDED", "FAILED"}


class NotificationProjectionService:
    async def project(self, *, uow, outbox) -> None:
        if uow.notifications is None:
            raise RuntimeError("Notification Repository 未初始化")
        envelope = NotificationEnvelope.model_validate(outbox.payload_json)
        definition = event_definition(envelope.event_type)
        if outbox.producer_service != definition.producer_service:
            raise ValueError("NOTIFICATION_EVENT_PRODUCER_INVALID")
        if int(outbox.domain_id) != envelope.domain_id:
            raise ValueError("NOTIFICATION_EVENT_DOMAIN_MISMATCH")

        operation = await self._project_operation(
            repository=uow.notifications,
            outbox=outbox,
            envelope=envelope,
            definition=definition,
        )
        recipients = set(envelope.recipient_actor_ids)
        if operation is not None and definition.operation_status in _TERMINAL:
            recipients.update(await uow.notifications.watcher_actor_ids(
                operation_id=operation.operation_id,
                domain_id=envelope.domain_id,
            ))
        title = definition.title
        if envelope.resource_name:
            title = f"{title}：{envelope.resource_name}"[:300]

        for actor_id in sorted(recipients):
            if definition.notify and await uow.notifications.preference_enabled(
                domain_id=envelope.domain_id,
                actor_id=actor_id,
                event_type=envelope.event_type,
            ):
                if not await uow.notifications.inbox_exists(
                    outbox_id=outbox.outbox_id, actor_id=actor_id,
                ):
                    await uow.notifications.add_inbox(NotificationInboxEntity(
                        outbox_id=outbox.outbox_id,
                        domain_id=envelope.domain_id,
                        recipient_actor_id=actor_id,
                        event_type=envelope.event_type,
                        category=definition.category,
                        severity=definition.severity,
                        title=title,
                        summary=envelope.summary,
                        resource_type=envelope.resource_type,
                        resource_id=envelope.resource_id,
                        operation_id=(operation.operation_id if operation else None),
                        expires_at=envelope.occurred_at + timedelta(
                            days=definition.retention_days
                        ),
                    ))
            await self._project_work_item(
                repository=uow.notifications,
                outbox=outbox,
                envelope=envelope,
                definition=definition,
                actor_id=actor_id,
                title=title,
            )

    @staticmethod
    async def _project_operation(
        *, repository, outbox, envelope, definition,
    ) -> BackgroundOperationEntity | None:
        if definition.operation_status is None or envelope.operation_id is None:
            return None
        operation = await repository.operation_for_update(
            producer=outbox.producer_service,
            source_operation_id=envelope.operation_id,
        )
        if operation is None:
            operation = BackgroundOperationEntity(
                domain_id=envelope.domain_id,
                producer_service=outbox.producer_service,
                source_operation_id=envelope.operation_id,
                initiator_actor_id=envelope.initiator_actor_id,
                resource_type=envelope.resource_type,
                resource_id=envelope.resource_id,
                resource_name=envelope.resource_name,
                status=definition.operation_status,
                progress_current=envelope.safe_data.get("progress_current"),
                progress_total=envelope.safe_data.get("progress_total"),
                error_code=envelope.safe_data.get("error_code"),
                summary=envelope.summary,
                last_outbox_id=outbox.outbox_id,
                last_occurred_at=envelope.occurred_at,
            )
            await repository.add_operation(operation)
            return operation
        if operation.domain_id != envelope.domain_id:
            raise ValueError("NOTIFICATION_OPERATION_DOMAIN_MISMATCH")
        if envelope.occurred_at < operation.last_occurred_at:
            return operation
        operation.status = definition.operation_status
        operation.resource_name = envelope.resource_name
        operation.progress_current = envelope.safe_data.get("progress_current")
        operation.progress_total = envelope.safe_data.get("progress_total")
        operation.error_code = envelope.safe_data.get("error_code")
        operation.summary = envelope.summary
        operation.last_outbox_id = outbox.outbox_id
        operation.last_occurred_at = envelope.occurred_at
        operation.row_version = int(operation.row_version) + 1
        return operation

    @staticmethod
    async def _project_work_item(
        *, repository, outbox, envelope, definition,
        actor_id: str, title: str,
    ) -> None:
        action_type = definition.work_item_action
        if action_type is None:
            return
        item = await repository.work_item_for_update(
            domain_id=envelope.domain_id,
            actor_id=actor_id,
            resource_type=envelope.resource_type,
            resource_id=envelope.resource_id,
            action_type=action_type,
        )
        if definition.resolve_work_item:
            if item is None:
                item = WorkItemEntity(
                    domain_id=envelope.domain_id,
                    actor_id=actor_id,
                    resource_type=envelope.resource_type,
                    resource_id=envelope.resource_id,
                    action_type=action_type,
                    title=title,
                    summary=envelope.summary,
                    status="COMPLETED",
                    opened_outbox_id=outbox.outbox_id,
                    resolved_outbox_id=outbox.outbox_id,
                    last_occurred_at=envelope.occurred_at,
                )
                await repository.add_work_item(item)
                return
            if (
                item.status == "OPEN"
                and envelope.occurred_at >= item.last_occurred_at
            ):
                item.status = "COMPLETED"
                item.resolved_outbox_id = outbox.outbox_id
                item.last_occurred_at = envelope.occurred_at
                item.row_version = int(item.row_version) + 1
            return
        if item is None:
            await repository.add_work_item(WorkItemEntity(
                domain_id=envelope.domain_id,
                actor_id=actor_id,
                resource_type=envelope.resource_type,
                resource_id=envelope.resource_id,
                action_type=action_type,
                title=title,
                summary=envelope.summary,
                opened_outbox_id=outbox.outbox_id,
                last_occurred_at=envelope.occurred_at,
            ))
        elif envelope.occurred_at >= item.last_occurred_at:
            item.status = "OPEN"
            item.resolved_outbox_id = None
            item.title = title
            item.summary = envelope.summary
            item.opened_outbox_id = outbox.outbox_id
            item.last_occurred_at = envelope.occurred_at
            item.row_version = int(item.row_version) + 1


class NotificationDispatcher:
    """短租约领取；每条事件使用独立 UoW 投影与隔离。"""

    def __init__(
        self, *, uow_factory, batch_size: int = 50,
        lease_seconds: int = 60, max_attempts: int = 5,
    ) -> None:
        self._uow_factory = uow_factory
        self._batch_size = batch_size
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._projection = NotificationProjectionService()

    async def dispatch_once(self) -> int:
        async with self._uow_factory() as uow:
            if uow.notifications is None:
                raise RuntimeError("Notification Repository 未初始化")
            await uow.notifications.recover_expired(
                now=datetime.now(timezone.utc)
            )
            claimed = await uow.notifications.claim_batch(
                limit=self._batch_size,
                lease_seconds=self._lease_seconds,
            )
            await uow.commit()
        for outbox_id, lease_token in claimed:
            try:
                async with self._uow_factory() as uow:
                    if uow.notifications is None:
                        raise RuntimeError("Notification Repository 未初始化")
                    row = await uow.notifications.claimed(
                        outbox_id=outbox_id, lease_token=lease_token,
                    )
                    if row is None:
                        continue
                    await self._projection.project(uow=uow, outbox=row)
                    await uow.notifications.mark_published(row)
                    await uow.commit()
            except Exception as exc:
                error_code = type(exc).__name__.upper()[:128]
                async with self._uow_factory() as uow:
                    if uow.notifications is None:
                        raise RuntimeError("Notification Repository 未初始化")
                    status = await uow.notifications.release_failed(
                        outbox_id=outbox_id,
                        lease_token=lease_token,
                        error_code=error_code,
                        max_attempts=self._max_attempts,
                    )
                    await uow.commit()
                logger.warning(
                    "通知事件投影失败 | outbox_id={} | status={} | error_code={}",
                    outbox_id, status, error_code,
                )
        return len(claimed)
