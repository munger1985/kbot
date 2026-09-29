"""工作负载快照与活动采样 Repository。"""

from collections.abc import Callable, Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    ActivitySampleEntity,
    WorkloadMetricEntity,
    WorkloadSnapshotEntity,
    WorkloadStatementEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class WorkloadRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(session, assert_active)

    async def add_snapshot_bundle(
        self,
        *,
        snapshot: WorkloadSnapshotEntity,
        statements: Sequence[WorkloadStatementEntity] = (),
        metrics: Sequence[WorkloadMetricEntity] = (),
    ) -> WorkloadSnapshotEntity:
        self._check_active()
        self._session.add(snapshot)
        self._session.add_all([*statements, *metrics])
        await self._session.flush()
        return snapshot

    async def get_snapshot_scoped(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        workload_snapshot_id: UUID,
    ) -> WorkloadSnapshotEntity | None:
        self._check_active()
        statement = select(WorkloadSnapshotEntity).where(
            WorkloadSnapshotEntity.domain_id == domain_id,
            WorkloadSnapshotEntity.target_id == target_id,
            WorkloadSnapshotEntity.workload_snapshot_id == workload_snapshot_id,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_snapshot_by_schedule(
        self, *, domain_id: int, target_id: UUID, scheduled_for: datetime
    ) -> WorkloadSnapshotEntity | None:
        self._check_active()
        statement = select(WorkloadSnapshotEntity).where(
            WorkloadSnapshotEntity.domain_id == domain_id,
            WorkloadSnapshotEntity.target_id == target_id,
            WorkloadSnapshotEntity.scheduled_for == scheduled_for,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_snapshots(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        period_start: datetime,
        period_end: datetime,
    ) -> list[WorkloadSnapshotEntity]:
        self._check_active()
        statement = (
            select(WorkloadSnapshotEntity)
            .where(
                WorkloadSnapshotEntity.domain_id == domain_id,
                WorkloadSnapshotEntity.target_id == target_id,
                WorkloadSnapshotEntity.collected_at >= period_start,
                WorkloadSnapshotEntity.collected_at < period_end,
            )
            .order_by(
                WorkloadSnapshotEntity.collected_at,
                WorkloadSnapshotEntity.workload_snapshot_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_statements(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        workload_snapshot_ids: Sequence[UUID],
    ) -> list[WorkloadStatementEntity]:
        self._check_active()
        if not workload_snapshot_ids:
            return []
        statement = (
            select(WorkloadStatementEntity)
            .where(
                WorkloadStatementEntity.domain_id == domain_id,
                WorkloadStatementEntity.target_id == target_id,
                WorkloadStatementEntity.workload_snapshot_id.in_(
                    tuple(workload_snapshot_ids)
                ),
            )
            .order_by(
                WorkloadStatementEntity.workload_snapshot_id,
                WorkloadStatementEntity.rank_no,
                WorkloadStatementEntity.workload_statement_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_metrics(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        workload_snapshot_ids: Sequence[UUID],
    ) -> list[WorkloadMetricEntity]:
        self._check_active()
        if not workload_snapshot_ids:
            return []
        statement = (
            select(WorkloadMetricEntity)
            .where(
                WorkloadMetricEntity.domain_id == domain_id,
                WorkloadMetricEntity.target_id == target_id,
                WorkloadMetricEntity.workload_snapshot_id.in_(
                    tuple(workload_snapshot_ids)
                ),
            )
            .order_by(
                WorkloadMetricEntity.workload_snapshot_id,
                WorkloadMetricEntity.metric_family,
                WorkloadMetricEntity.metric_code,
                WorkloadMetricEntity.dimension_key,
                WorkloadMetricEntity.workload_metric_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def add_activity_samples(
        self, samples: Sequence[ActivitySampleEntity]
    ) -> None:
        self._check_active()
        self._session.add_all(list(samples))
        await self._session.flush()

    async def list_activity_samples(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        period_start: datetime,
        period_end: datetime,
        limit: int,
    ) -> list[ActivitySampleEntity]:
        self._check_active()
        statement = (
            select(ActivitySampleEntity)
            .where(
                ActivitySampleEntity.domain_id == domain_id,
                ActivitySampleEntity.target_id == target_id,
                ActivitySampleEntity.sampled_at >= period_start,
                ActivitySampleEntity.sampled_at < period_end,
            )
            .order_by(
                ActivitySampleEntity.sampled_at,
                ActivitySampleEntity.activity_sample_id,
            )
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def delete_expired(
        self, *, now: datetime, limit: int = 1000
    ) -> dict[str, int]:
        """按明确主键小批清理，避免一次删除扩大锁范围。"""
        self._check_active()
        snapshot_ids = tuple(
            (
                await self._session.execute(
                    select(WorkloadSnapshotEntity.workload_snapshot_id)
                    .where(WorkloadSnapshotEntity.expires_at < now)
                    .order_by(
                        WorkloadSnapshotEntity.expires_at,
                        WorkloadSnapshotEntity.workload_snapshot_id,
                    )
                    .limit(limit)
                )
            ).scalars()
        )
        sample_ids = tuple(
            (
                await self._session.execute(
                    select(ActivitySampleEntity.activity_sample_id)
                    .where(ActivitySampleEntity.expires_at < now)
                    .order_by(
                        ActivitySampleEntity.expires_at,
                        ActivitySampleEntity.activity_sample_id,
                    )
                    .limit(limit)
                )
            ).scalars()
        )
        if snapshot_ids:
            await self._session.execute(
                delete(WorkloadStatementEntity).where(
                    WorkloadStatementEntity.workload_snapshot_id.in_(snapshot_ids)
                )
            )
            await self._session.execute(
                delete(WorkloadMetricEntity).where(
                    WorkloadMetricEntity.workload_snapshot_id.in_(snapshot_ids)
                )
            )
            await self._session.execute(
                delete(WorkloadSnapshotEntity).where(
                    WorkloadSnapshotEntity.workload_snapshot_id.in_(snapshot_ids)
                )
            )
        if sample_ids:
            await self._session.execute(
                delete(ActivitySampleEntity).where(
                    ActivitySampleEntity.activity_sample_id.in_(sample_ids)
                )
            )
        return {"snapshots": len(snapshot_ids), "samples": len(sample_ids)}
