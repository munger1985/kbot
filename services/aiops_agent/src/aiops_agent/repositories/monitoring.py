"""诊断源、信号事件与故障情境的 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    DiagnosticSourceEntity,
    SignalEventEntity,
    SituationEntity,
    SituationEventEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class DiagnosticSourceRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add(
        self, entity: DiagnosticSourceEntity
    ) -> DiagnosticSourceEntity:
        return await self._add(entity)

    async def delete_source(self, entity: DiagnosticSourceEntity) -> None:
        """仅由用例层在确认停用后删除无关联监控源。"""
        self._check_active()
        await self._session.delete(entity)
        await self._session.flush()

    async def get_scoped(
        self,
        *,
        diagnostic_source_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> DiagnosticSourceEntity | None:
        self._check_active()
        statement: Select = select(DiagnosticSourceEntity).where(
            DiagnosticSourceEntity.diagnostic_source_id == diagnostic_source_id,
            DiagnosticSourceEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def page_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_updated_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[DiagnosticSourceEntity]:
        self._check_active()
        statement = select(DiagnosticSourceEntity).where(
            DiagnosticSourceEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(
                DiagnosticSourceEntity.status.in_(statuses)
            )
        if before_updated_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    DiagnosticSourceEntity.updated_at < before_updated_at,
                    and_(
                        DiagnosticSourceEntity.updated_at == before_updated_at,
                        DiagnosticSourceEntity.diagnostic_source_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            DiagnosticSourceEntity.updated_at.desc(),
            DiagnosticSourceEntity.diagnostic_source_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def claim_due_connectivity(
        self, *, due_before: datetime, pending_before: datetime
    ) -> DiagnosticSourceEntity | None:
        """锁定一个到期监控源，供多副本 Scheduler 安全发起检查。"""
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT DIAGNOSTIC_SOURCE_ID
                        FROM KBOT_OPS_DIAGNOSTIC_SOURCE
                        WHERE (
                            (
                                LAST_CONNECTIVITY_CHECK_AT IS NULL
                                AND (
                                    CONNECTIVITY_CHECK_REQUESTED_AT IS NULL
                                    OR CONNECTIVITY_CHECK_REQUESTED_AT
                                        <= :pending_before
                                )
                            )
                            OR (
                                LAST_CONNECTIVITY_CHECK_AT <= :due_before
                                AND CONNECTIVITY_STATUS <> 'CHECKING'
                            )
                            OR (
                                CONNECTIVITY_STATUS = 'CHECKING'
                                AND CONNECTIVITY_CHECK_REQUESTED_AT
                                    <= :pending_before
                            )
                        )
                        ORDER BY LAST_CONNECTIVITY_CHECK_AT NULLS FIRST,
                                 DIAGNOSTIC_SOURCE_ID
                        FOR UPDATE OF DIAGNOSTIC_SOURCE_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={
                "due_before": due_before,
                "pending_before": pending_before,
            },
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(DiagnosticSourceEntity).where(
                    DiagnosticSourceEntity.diagnostic_source_id == claimed_id
                )
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(f"领取后的监控源不存在：{claimed_id}")
        return entity

    async def update_config(
        self,
        *,
        diagnostic_source_id: UUID,
        domain_id: int,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": DiagnosticSourceEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(DiagnosticSourceEntity)
            .where(
                DiagnosticSourceEntity.diagnostic_source_id == diagnostic_source_id,
                DiagnosticSourceEntity.domain_id == domain_id,
                DiagnosticSourceEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def request_connectivity_check(
        self,
        *,
        diagnostic_source_id: UUID,
        domain_id: int,
        expected_version: int,
        request_id: UUID,
        requested_at: datetime,
        updated_by: str,
    ) -> bool:
        return await self.update_config(
            diagnostic_source_id=diagnostic_source_id,
            domain_id=domain_id,
            expected_version=expected_version,
            values={
                "connectivity_check_request_id": request_id,
                "connectivity_check_requested_at": requested_at,
                "updated_by": updated_by,
            },
        )

    async def get_by_webhook_hash(
        self,
        *,
        webhook_key_hash: str,
        now: datetime,
    ) -> DiagnosticSourceEntity | None:
        self._check_active()
        statement = select(DiagnosticSourceEntity).where(
            DiagnosticSourceEntity.status == "ENABLED",
            or_(
                DiagnosticSourceEntity.webhook_key_hash == webhook_key_hash,
                (
                    (
                        DiagnosticSourceEntity.previous_webhook_key_hash
                        == webhook_key_hash
                    )
                    & (
                        DiagnosticSourceEntity.previous_webhook_key_expires_at
                        > now
                    )
                ),
            ),
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def update_connectivity(
        self,
        *,
        diagnostic_source_id: UUID,
        connectivity_check_request_id: UUID,
        expected_config_version: int,
        expected_connectivity_version: int,
        connectivity_status: str,
        checked_at: datetime,
        last_error_code: str | None,
        discovered_capabilities: dict | None = None,
    ) -> bool:
        self._check_active()
        statement = (
            update(DiagnosticSourceEntity)
            .where(
                DiagnosticSourceEntity.diagnostic_source_id == diagnostic_source_id,
                DiagnosticSourceEntity.connectivity_check_request_id
                == connectivity_check_request_id,
                DiagnosticSourceEntity.row_version == expected_config_version,
                DiagnosticSourceEntity.connectivity_version
                == expected_connectivity_version,
            )
            .values(
                connectivity_status=connectivity_status,
                last_connectivity_check_at=checked_at,
                last_connectivity_success_at=(
                    checked_at
                    if connectivity_status in {"CONNECTED", "DEGRADED"}
                    else DiagnosticSourceEntity.last_connectivity_success_at
                ),
                last_error_code=last_error_code,
                discovered_capabilities_json=discovered_capabilities,
                connectivity_version=(
                    DiagnosticSourceEntity.connectivity_version + 1
                ),
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def reduce_connectivity(
        self,
        *,
        diagnostic_source_id: UUID,
        expected_config_version: int,
        expected_connectivity_version: int,
        connectivity_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        """归并普通采集连通性，不消费显式检查请求。"""
        self._check_active()
        statement = (
            update(DiagnosticSourceEntity)
            .where(
                DiagnosticSourceEntity.diagnostic_source_id
                == diagnostic_source_id,
                DiagnosticSourceEntity.row_version
                == expected_config_version,
                DiagnosticSourceEntity.connectivity_version
                == expected_connectivity_version,
            )
            .values(
                connectivity_status=connectivity_status,
                last_connectivity_check_at=checked_at,
                last_error_code=last_error_code,
                connectivity_version=(
                    DiagnosticSourceEntity.connectivity_version + 1
                ),
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class SituationRepository(AIOpsRepository):
    async def add_event(self, entity: SignalEventEntity) -> SignalEventEntity:
        return await self._add(entity)

    async def add_situation(self, entity: SituationEntity) -> SituationEntity:
        return await self._add(entity)

    async def add_situation_event(
        self, entity: SituationEventEntity
    ) -> SituationEventEntity:
        return await self._add(entity)

    async def get_situation(
        self, *, situation_id: UUID
    ) -> SituationEntity | None:
        self._check_active()
        return (
            await self._session.execute(
                select(SituationEntity).where(
                    SituationEntity.situation_id == situation_id
                )
            )
        ).scalar_one_or_none()

    async def get_situation_scoped(
        self, *, situation_id: UUID, domain_id: int
    ) -> SituationEntity | None:
        self._check_active()
        return (await self._session.execute(select(SituationEntity).where(
            SituationEntity.situation_id == situation_id,
            SituationEntity.domain_id == domain_id,
        ))).scalar_one_or_none()

    async def page_situations(
        self, *, domain_id: int, target_id: UUID | None = None,
        status: str | None = None, severity: str | None = None,
        before_created_at: datetime | None = None,
        before_id: UUID | None = None, limit: int = 51,
    ) -> list[SituationEntity]:
        self._check_active()
        statement = select(SituationEntity).where(SituationEntity.domain_id == domain_id)
        if target_id is not None:
            statement = statement.where(SituationEntity.target_id == target_id)
        if status is not None:
            statement = statement.where(SituationEntity.status == status)
        if severity is not None:
            statement = statement.where(SituationEntity.severity == severity)
        if before_created_at is not None and before_id is not None:
            statement = statement.where(or_(
                SituationEntity.created_at < before_created_at,
                (SituationEntity.created_at == before_created_at)
                & (SituationEntity.situation_id < before_id),
            ))
        statement = statement.order_by(
            SituationEntity.created_at.desc(), SituationEntity.situation_id.desc()
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def list_events_for_situation(
        self, *, situation_id: UUID, limit: int = 200
    ) -> list[SignalEventEntity]:
        self._check_active()
        statement = select(SignalEventEntity).join(
            SituationEventEntity,
            SituationEventEntity.signal_event_id == SignalEventEntity.signal_event_id,
        ).where(SituationEventEntity.situation_id == situation_id).order_by(
            SignalEventEntity.occurred_at.desc(), SignalEventEntity.signal_event_id.desc()
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def summarize_sources_for_situation(
        self, *, situation_id: UUID
    ) -> list[dict]:
        """按监控来源聚合情境事件，并保留每个来源的最新告警内容。"""
        self._check_active()
        source_partition = SignalEventEntity.diagnostic_source_id
        ranked = (
            select(
                SignalEventEntity.diagnostic_source_id.label(
                    "diagnostic_source_id"
                ),
                DiagnosticSourceEntity.display_name.label("display_name"),
                DiagnosticSourceEntity.source_type.label("source_type"),
                SignalEventEntity.event_class.label("latest_event_class"),
                SignalEventEntity.severity.label("latest_severity"),
                SignalEventEntity.normalized_status.label("latest_status"),
                SignalEventEntity.summary.label("latest_summary"),
                func.count(SignalEventEntity.signal_event_id)
                .over(partition_by=source_partition)
                .label("event_count"),
                func.min(SignalEventEntity.occurred_at)
                .over(partition_by=source_partition)
                .label("first_observed_at"),
                func.max(SignalEventEntity.occurred_at)
                .over(partition_by=source_partition)
                .label("last_observed_at"),
                func.row_number()
                .over(
                    partition_by=source_partition,
                    order_by=(
                        SignalEventEntity.occurred_at.desc(),
                        SignalEventEntity.signal_event_id.desc(),
                    ),
                )
                .label("source_rank"),
            )
            .select_from(SituationEventEntity)
            .join(
                SignalEventEntity,
                SignalEventEntity.signal_event_id
                == SituationEventEntity.signal_event_id,
            )
            .join(
                DiagnosticSourceEntity,
                DiagnosticSourceEntity.diagnostic_source_id
                == SignalEventEntity.diagnostic_source_id,
            )
            .where(SituationEventEntity.situation_id == situation_id)
            .subquery()
        )
        statement = (
            select(ranked)
            .where(ranked.c.source_rank == 1)
            .order_by(
                ranked.c.last_observed_at.desc(),
                ranked.c.diagnostic_source_id,
            )
        )
        rows = (await self._session.execute(statement)).mappings().all()
        return [dict(row) for row in rows]

    async def get_event_by_source(
        self,
        *,
        diagnostic_source_id: UUID,
        source_event_key: str,
    ) -> SignalEventEntity | None:
        self._check_active()
        statement = select(SignalEventEntity).where(
            SignalEventEntity.diagnostic_source_id == diagnostic_source_id,
            SignalEventEntity.source_event_key == source_event_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_signal_event_ids_by_inbox(
        self, *, inbox_id: UUID
    ) -> list[UUID]:
        self._check_active()
        statement = (
            select(SignalEventEntity.signal_event_id)
            .where(SignalEventEntity.source_inbox_id == inbox_id)
            .order_by(
                SignalEventEntity.created_at,
                SignalEventEntity.signal_event_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_active_situation(
        self,
        *,
        target_id: UUID,
        correlation_hash: str,
        lock: bool = False,
    ) -> SituationEntity | None:
        self._check_active()
        statement: Select = select(SituationEntity).where(
            SituationEntity.target_id == target_id,
            SituationEntity.correlation_hash == correlation_hash,
            SituationEntity.status.in_(
                (
                    "OPEN",
                    "ACKNOWLEDGED",
                    "INVESTIGATING",
                    "DIAGNOSED",
                    "MITIGATING",
                    "OBSERVING",
                    "SUPPRESSED",
                )
            ),
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def has_open_signal_state(self, *, situation_id: UUID) -> bool:
        """按来源内关联键的最新事件判断 Situation 是否仍有未恢复信号。"""
        self._check_active()
        ranked = (
            select(
                SignalEventEntity.normalized_status.label("signal_status"),
                func.row_number()
                .over(
                    partition_by=SignalEventEntity.dedup_hash,
                    order_by=(
                        SignalEventEntity.occurred_at.desc(),
                        SignalEventEntity.signal_event_id.desc(),
                    ),
                )
                .label("signal_rank"),
            )
            .join(
                SituationEventEntity,
                SituationEventEntity.signal_event_id
                == SignalEventEntity.signal_event_id,
            )
            .where(SituationEventEntity.situation_id == situation_id)
            .subquery()
        )
        statement = (
            select(ranked.c.signal_status)
            .where(
                ranked.c.signal_rank == 1,
                ranked.c.signal_status.in_(("OPEN", "UPDATED")),
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalar_one_or_none() is not None

    async def attach_event(
        self,
        *,
        signal_event_id: UUID,
        situation_id: UUID,
        expected_statuses: Collection[str] = ("RECEIVED",),
        processing_status: str = "CORRELATED",
    ) -> bool:
        self._check_active()
        statement = (
            update(SignalEventEntity)
            .where(
                SignalEventEntity.signal_event_id == signal_event_id,
                SignalEventEntity.processing_status.in_(expected_statuses),
            )
            .values(processing_status=processing_status)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_situation(
        self,
        *,
        situation_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        status: str,
        severity: str,
        title: str,
        last_observed_at: datetime,
        correlation_json: dict | None,
        resolved_at: datetime | None = None,
        increment_event_count: bool = True,
    ) -> bool:
        self._check_active()
        values = {
            "status": status,
            "severity": severity,
            "title": title,
            "last_observed_at": last_observed_at,
            "correlation_json": correlation_json,
            "resolved_at": resolved_at,
            "row_version": SituationEntity.row_version + 1,
            "updated_at": datetime.now(UTC),
        }
        if increment_event_count:
            values["event_count"] = SituationEntity.event_count + 1
        statement = (
            update(SituationEntity)
            .where(
                SituationEntity.situation_id == situation_id,
                SituationEntity.row_version == expected_version,
                SituationEntity.status.in_(allowed_statuses),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
