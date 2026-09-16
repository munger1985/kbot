"""智能工作台 X Search 与文生图 Repository。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.identity import uuid7

from assistant_app.entities import (
    AssistantMediaAssetEntity,
    AssistantModelBindingEntity,
    AssistantPromptRevisionEntity,
    AssistantRunEntity,
    AssistantRunEventEntity,
    AssistantXSourceEntity,
)


IN_PROGRESS_STATUSES = (
    "SEARCHING",
    "ORGANIZING_SOURCES",
    "COMPOSING",
    "GENERATING",
)


class AssistantModelBindingRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantModelBindingEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get_by_role(
        self, *, domain_id: int, role: str, lock: bool = False,
    ) -> AssistantModelBindingEntity | None:
        statement = select(AssistantModelBindingEntity).where(
            AssistantModelBindingEntity.domain_id == domain_id,
            AssistantModelBindingEntity.role == role,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id: int) -> list[AssistantModelBindingEntity]:
        rows = await self._session.scalars(
            select(AssistantModelBindingEntity)
            .where(AssistantModelBindingEntity.domain_id == domain_id)
            .order_by(AssistantModelBindingEntity.role)
        )
        return list(rows)


class AssistantRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantRunEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, run_id: UUID, lock: bool = False,
    ) -> AssistantRunEntity | None:
        statement = select(AssistantRunEntity).where(
            AssistantRunEntity.domain_id == domain_id,
            AssistantRunEntity.run_id == run_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self, *, domain_id: int, actor_id: str, idempotency_key: str,
    ) -> AssistantRunEntity | None:
        return (
            await self._session.execute(
                select(AssistantRunEntity).where(
                    AssistantRunEntity.domain_id == domain_id,
                    AssistantRunEntity.actor_id == actor_id,
                    AssistantRunEntity.idempotency_key == idempotency_key,
                )
            )
        ).scalar_one_or_none()

    async def list(
        self,
        *,
        domain_id: int,
        kind: str | None = None,
        status: str | None = None,
        actor_id: str | None = None,
        limit: int = 50,
    ) -> list[AssistantRunEntity]:
        conditions = [AssistantRunEntity.domain_id == domain_id]
        if kind:
            conditions.append(AssistantRunEntity.kind == kind)
        if status:
            conditions.append(AssistantRunEntity.status == status)
        if actor_id:
            conditions.append(AssistantRunEntity.actor_id == actor_id)
        rows = await self._session.scalars(
            select(AssistantRunEntity)
            .where(*conditions)
            .order_by(AssistantRunEntity.created_at.desc(), AssistantRunEntity.run_id)
            .limit(limit)
        )
        return list(rows)

    async def claim(self, *, worker_id: str, lease_until: datetime) -> AssistantRunEntity | None:
        now = datetime.now(timezone.utc)
        eligibility = (
            or_(
                (
                    (AssistantRunEntity.status == "ACCEPTED")
                    & or_(
                        AssistantRunEntity.lease_until.is_(None),
                        AssistantRunEntity.lease_until < now,
                    )
                ),
                (
                    AssistantRunEntity.status.in_(IN_PROGRESS_STATUSES)
                    & AssistantRunEntity.lease_until.is_not(None)
                    & (AssistantRunEntity.lease_until < now)
                ),
            ),
        )
        # Oracle 不允许 FETCH FIRST 与 FOR UPDATE 作用于同一查询块。
        candidate_ids = list(
            (
                await self._session.execute(
                    select(AssistantRunEntity.run_id)
                    .where(*eligibility)
                    .order_by(AssistantRunEntity.created_at, AssistantRunEntity.run_id)
                    .limit(32)
                )
            ).scalars()
        )
        for run_id in candidate_ids:
            row = (
                await self._session.execute(
                    select(AssistantRunEntity)
                    .where(AssistantRunEntity.run_id == run_id, *eligibility)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if row is None:
                continue
            row.lease_owner = worker_id
            row.lease_token = uuid7()
            row.lease_until = lease_until
            row.row_version = int(row.row_version) + 1
            if row.started_at is None:
                row.started_at = now
            await self._session.flush()
            return row
        return None

    async def delete(self, row: AssistantRunEntity) -> None:
        await self._session.delete(row)
        await self._session.flush()


class AssistantRunEventRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantRunEventEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int, run_id: UUID) -> list[AssistantRunEventEntity]:
        rows = await self._session.scalars(
            select(AssistantRunEventEntity)
            .where(
                AssistantRunEventEntity.domain_id == domain_id,
                AssistantRunEventEntity.run_id == run_id,
            )
            .order_by(AssistantRunEventEntity.sequence_no)
        )
        return list(rows)

    async def next_sequence(self, *, run_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(AssistantRunEventEntity.sequence_no)).where(
                AssistantRunEventEntity.run_id == run_id
            )
        )
        return int(value or 0) + 1

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(AssistantRunEventEntity).where(
                AssistantRunEventEntity.domain_id == domain_id,
                AssistantRunEventEntity.run_id == run_id,
            )
        )


class AssistantXSourceRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantXSourceEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int, run_id: UUID) -> list[AssistantXSourceEntity]:
        rows = await self._session.scalars(
            select(AssistantXSourceEntity)
            .where(
                AssistantXSourceEntity.domain_id == domain_id,
                AssistantXSourceEntity.run_id == run_id,
            )
            .order_by(AssistantXSourceEntity.display_order)
        )
        return list(rows)

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(AssistantXSourceEntity).where(
                AssistantXSourceEntity.domain_id == domain_id,
                AssistantXSourceEntity.run_id == run_id,
            )
        )


class AssistantPromptRevisionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantPromptRevisionEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list_by_run(
        self, *, domain_id: int, run_id: UUID,
    ) -> list[AssistantPromptRevisionEntity]:
        rows = await self._session.scalars(
            select(AssistantPromptRevisionEntity)
            .where(
                AssistantPromptRevisionEntity.domain_id == domain_id,
                AssistantPromptRevisionEntity.run_id == run_id,
            )
            .order_by(AssistantPromptRevisionEntity.version_no)
        )
        return list(rows)

    async def get(
        self, *, prompt_revision_id: UUID,
    ) -> AssistantPromptRevisionEntity | None:
        return (
            await self._session.execute(
                select(AssistantPromptRevisionEntity).where(
                    AssistantPromptRevisionEntity.prompt_revision_id == prompt_revision_id
                )
            )
        ).scalar_one_or_none()

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(AssistantPromptRevisionEntity).where(
                AssistantPromptRevisionEntity.domain_id == domain_id,
                AssistantPromptRevisionEntity.run_id == run_id,
            )
        )


class AssistantMediaAssetRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: AssistantMediaAssetEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, asset_id: UUID,
    ) -> AssistantMediaAssetEntity | None:
        return (
            await self._session.execute(
                select(AssistantMediaAssetEntity).where(
                    AssistantMediaAssetEntity.domain_id == domain_id,
                    AssistantMediaAssetEntity.asset_id == asset_id,
                )
            )
        ).scalar_one_or_none()

    async def list(
        self,
        *,
        domain_id: int,
        run_id: UUID | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[AssistantMediaAssetEntity]:
        conditions = [AssistantMediaAssetEntity.domain_id == domain_id]
        if run_id is not None:
            conditions.append(AssistantMediaAssetEntity.run_id == run_id)
        if status:
            conditions.append(AssistantMediaAssetEntity.status == status)
        rows = await self._session.scalars(
            select(AssistantMediaAssetEntity)
            .where(*conditions)
            .order_by(
                AssistantMediaAssetEntity.created_at.desc(),
                AssistantMediaAssetEntity.asset_id,
            )
            .limit(limit)
        )
        return list(rows)

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(AssistantMediaAssetEntity).where(
                AssistantMediaAssetEntity.domain_id == domain_id,
                AssistantMediaAssetEntity.run_id == run_id,
            )
        )
