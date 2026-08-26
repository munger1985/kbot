"""Target 站内主动分享订阅用例。"""

from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import (
    resource_not_found,
    row_version_changed,
)
from aiops_agent.entities import NotificationSubscriptionEntity
from platform_core.contracts.aiops import (
    NotificationSubscriptionList,
    NotificationSubscriptionUpsert,
    NotificationSubscriptionView,
)
from platform_core.identity import uuid7


class NotificationConfigurationMixin:
    async def list_notification_subscriptions(
        self, *, scope
    ) -> NotificationSubscriptionList:
        async with self._uow_factory() as uow:
            rows = await uow.notification_subscriptions.list_for_actor(
                domain_id=scope.domain_id,
                actor_id=scope.actor_id,
            )
            return NotificationSubscriptionList(
                items=tuple(self._notification_view(row) for row in rows)
            )

    async def upsert_notification_subscription(
        self,
        *,
        scope,
        target_id: UUID,
        request: NotificationSubscriptionUpsert,
        expected_version: int | None,
    ) -> NotificationSubscriptionView:
        async with self._uow_factory() as uow:
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=scope.domain_id,
            )
            if target is None or target.status != "ENABLED":
                raise resource_not_found("Target")
            row = await uow.notification_subscriptions.get_for_actor(
                domain_id=scope.domain_id,
                target_id=target_id,
                actor_id=scope.actor_id,
                lock=True,
            )
            now = datetime.now(UTC)
            if row is None:
                if expected_version is not None:
                    raise row_version_changed()
                row = NotificationSubscriptionEntity(
                    subscription_id=uuid7(),
                    domain_id=scope.domain_id,
                    target_id=target_id,
                    recipient_actor_id=scope.actor_id,
                    channel="IN_APP",
                    minimum_severity=request.minimum_severity,
                    stages_json=list(request.stages),
                    status="ACTIVE",
                    row_version=1,
                    created_at=now,
                    updated_at=now,
                )
                await uow.notification_subscriptions.add(row)
            else:
                if expected_version is None or int(row.row_version) != expected_version:
                    raise row_version_changed()
                row.minimum_severity = request.minimum_severity
                row.stages_json = list(request.stages)
                row.status = "ACTIVE"
                row.row_version = int(row.row_version) + 1
                row.updated_at = now
            await uow.commit()
            return self._notification_view(row)

    async def disable_notification_subscription(
        self,
        *,
        scope,
        target_id: UUID,
        expected_version: int | None,
    ) -> None:
        async with self._uow_factory() as uow:
            row = await uow.notification_subscriptions.get_for_actor(
                domain_id=scope.domain_id,
                target_id=target_id,
                actor_id=scope.actor_id,
                lock=True,
            )
            if row is None:
                raise resource_not_found("主动分享订阅")
            if expected_version is None or int(row.row_version) != expected_version:
                raise row_version_changed()
            row.status = "DISABLED"
            row.row_version = int(row.row_version) + 1
            row.updated_at = datetime.now(UTC)
            await uow.commit()

    @staticmethod
    def _notification_view(row) -> NotificationSubscriptionView:
        return NotificationSubscriptionView(
            subscription_id=row.subscription_id,
            target_id=row.target_id,
            recipient_actor_id=row.recipient_actor_id,
            channel=row.channel,
            minimum_severity=row.minimum_severity,
            stages=tuple(row.stages_json or ()),
            status=row.status,
            row_version=int(row.row_version),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
