"""监控源、外部 Event 与 Alert 聚合的 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    MonitorSourceEntity,
    OpsAlertEntity,
    OpsEventEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class MonitorSourceRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add(
        self, entity: MonitorSourceEntity
    ) -> MonitorSourceEntity:
        return await self._add(entity)

    async def get_scoped(
        self,
        *,
        monitor_source_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> MonitorSourceEntity | None:
        self._check_active()
        statement: Select = select(MonitorSourceEntity).where(
            MonitorSourceEntity.monitor_source_id == monitor_source_id,
            MonitorSourceEntity.app_id == app_id,
            MonitorSourceEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_webhook_hash(
        self,
        *,
        webhook_key_hash: str,
        now: datetime,
    ) -> MonitorSourceEntity | None:
        self._check_active()
        statement = select(MonitorSourceEntity).where(
            MonitorSourceEntity.status == "ACTIVE",
            or_(
                MonitorSourceEntity.webhook_key_hash == webhook_key_hash,
                (
                    (
                        MonitorSourceEntity.previous_webhook_key_hash
                        == webhook_key_hash
                    )
                    & (
                        MonitorSourceEntity.previous_webhook_key_expires_at
                        > now
                    )
                ),
            ),
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def update_health(
        self,
        *,
        monitor_source_id: UUID,
        expected_health_version: int,
        health_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(MonitorSourceEntity)
            .where(
                MonitorSourceEntity.monitor_source_id == monitor_source_id,
                MonitorSourceEntity.health_version
                == expected_health_version,
            )
            .values(
                health_status=health_status,
                last_health_check_at=checked_at,
                last_error_code=last_error_code,
                health_version=MonitorSourceEntity.health_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class AlertRepository(AIOpsRepository):
    async def add_event(self, entity: OpsEventEntity) -> OpsEventEntity:
        return await self._add(entity)

    async def add_alert(self, entity: OpsAlertEntity) -> OpsAlertEntity:
        return await self._add(entity)

    async def get_event_by_source(
        self,
        *,
        monitor_source_id: UUID,
        source_event_key: str,
    ) -> OpsEventEntity | None:
        self._check_active()
        statement = select(OpsEventEntity).where(
            OpsEventEntity.monitor_source_id == monitor_source_id,
            OpsEventEntity.source_event_key == source_event_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active_alert(
        self,
        *,
        target_id: UUID,
        fingerprint: str,
        lock: bool = False,
    ) -> OpsAlertEntity | None:
        self._check_active()
        statement: Select = select(OpsAlertEntity).where(
            OpsAlertEntity.target_id == target_id,
            OpsAlertEntity.fingerprint == fingerprint,
            OpsAlertEntity.status.in_(
                ("OPEN", "ACKNOWLEDGED", "SUPPRESSED")
            ),
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def correlate_event(
        self,
        *,
        event_id: UUID,
        alert_id: UUID | None,
        expected_statuses: Collection[str] = ("RECEIVED",),
        processing_status: str = "CORRELATED",
    ) -> bool:
        self._check_active()
        statement = (
            update(OpsEventEntity)
            .where(
                OpsEventEntity.event_id == event_id,
                OpsEventEntity.processing_status.in_(expected_statuses),
            )
            .values(
                alert_id=alert_id,
                processing_status=processing_status,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_alert(
        self,
        *,
        alert_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        status: str,
        severity: str,
        summary: str,
        last_seen_at: datetime,
        correlation_json: dict | None,
        resolved_at: datetime | None = None,
        increment_event_count: bool = True,
    ) -> bool:
        self._check_active()
        values = {
            "status": status,
            "severity": severity,
            "summary": summary,
            "last_seen_at": last_seen_at,
            "correlation_json": correlation_json,
            "resolved_at": resolved_at,
            "row_version": OpsAlertEntity.row_version + 1,
            "updated_at": datetime.now(UTC),
        }
        if increment_event_count:
            values["event_count"] = OpsAlertEntity.event_count + 1
        statement = (
            update(OpsAlertEntity)
            .where(
                OpsAlertEntity.alert_id == alert_id,
                OpsAlertEntity.row_version == expected_version,
                OpsAlertEntity.status.in_(allowed_statuses),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
