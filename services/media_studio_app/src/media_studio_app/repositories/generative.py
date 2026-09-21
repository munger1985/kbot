"""多媒体创作工作台生成运行时 Repository。"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from platform_core.identity import uuid7

from media_studio_app.entities import (
    MediaStudioMediaAssetEntity,
    MediaStudioModelBindingEntity,
    MediaStudioPromptRevisionEntity,
    MediaStudioRunEntity,
    MediaStudioRunEventEntity,
)


IN_PROGRESS_STATUSES = (
    "GENERATING",
)

class MediaStudioModelBindingRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: MediaStudioModelBindingEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get_by_role(
        self, *, domain_id: int, role: str, lock: bool = False,
    ) -> MediaStudioModelBindingEntity | None:
        statement = select(MediaStudioModelBindingEntity).where(
            MediaStudioModelBindingEntity.domain_id == domain_id,
            MediaStudioModelBindingEntity.role == role,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list(self, *, domain_id: int) -> list[MediaStudioModelBindingEntity]:
        rows = await self._session.scalars(
            select(MediaStudioModelBindingEntity)
            .where(MediaStudioModelBindingEntity.domain_id == domain_id)
            .order_by(MediaStudioModelBindingEntity.role)
        )
        return list(rows)

    async def model_references(
        self, *, model_id: UUID
    ) -> list[MediaStudioModelBindingEntity]:
        rows = await self._session.scalars(
            select(MediaStudioModelBindingEntity).where(
                MediaStudioModelBindingEntity.model_id == model_id
            )
        )
        return list(rows)

class MediaStudioRunRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: MediaStudioRunEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, run_id: UUID, lock: bool = False,
    ) -> MediaStudioRunEntity | None:
        statement = select(MediaStudioRunEntity).where(
            MediaStudioRunEntity.domain_id == domain_id,
            MediaStudioRunEntity.run_id == run_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self, *, domain_id: int, actor_id: str, idempotency_key: str,
    ) -> MediaStudioRunEntity | None:
        return (
            await self._session.execute(
                select(MediaStudioRunEntity).where(
                    MediaStudioRunEntity.domain_id == domain_id,
                    MediaStudioRunEntity.actor_id == actor_id,
                    MediaStudioRunEntity.idempotency_key == idempotency_key,
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
    ) -> list[MediaStudioRunEntity]:
        conditions = [MediaStudioRunEntity.domain_id == domain_id]
        if kind:
            conditions.append(MediaStudioRunEntity.kind == kind)
        if status:
            conditions.append(MediaStudioRunEntity.status == status)
        if actor_id:
            conditions.append(MediaStudioRunEntity.actor_id == actor_id)
        rows = await self._session.scalars(
            select(MediaStudioRunEntity)
            .where(*conditions)
            .order_by(MediaStudioRunEntity.created_at.desc(), MediaStudioRunEntity.run_id)
            .limit(limit)
        )
        return list(rows)

    async def claim(self, *, worker_id: str, lease_until: datetime) -> MediaStudioRunEntity | None:
        now = datetime.now(timezone.utc)
        eligibility = (
            or_(
                (
                    (MediaStudioRunEntity.status == "ACCEPTED")
                    & or_(
                        MediaStudioRunEntity.lease_until.is_(None),
                        MediaStudioRunEntity.lease_until < now,
                    )
                ),
                (
                    MediaStudioRunEntity.status.in_(IN_PROGRESS_STATUSES)
                    & MediaStudioRunEntity.lease_until.is_not(None)
                    & (MediaStudioRunEntity.lease_until < now)
                ),
            ),
        )
        # Oracle 不允许 FETCH FIRST 与 FOR UPDATE 作用于同一查询块。
        candidate_ids = list(
            (
                await self._session.execute(
                    select(MediaStudioRunEntity.run_id)
                    .where(*eligibility)
                    .order_by(MediaStudioRunEntity.created_at, MediaStudioRunEntity.run_id)
                    .limit(32)
                )
            ).scalars()
        )
        for run_id in candidate_ids:
            row = (
                await self._session.execute(
                    select(MediaStudioRunEntity)
                    .where(MediaStudioRunEntity.run_id == run_id, *eligibility)
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

    async def delete(self, row: MediaStudioRunEntity) -> None:
        await self._session.delete(row)
        await self._session.flush()

class MediaStudioRunEventRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: MediaStudioRunEventEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list(self, *, domain_id: int, run_id: UUID) -> list[MediaStudioRunEventEntity]:
        rows = await self._session.scalars(
            select(MediaStudioRunEventEntity)
            .where(
                MediaStudioRunEventEntity.domain_id == domain_id,
                MediaStudioRunEventEntity.run_id == run_id,
            )
            .order_by(MediaStudioRunEventEntity.sequence_no)
        )
        return list(rows)

    async def next_sequence(self, *, run_id: UUID) -> int:
        value = await self._session.scalar(
            select(func.max(MediaStudioRunEventEntity.sequence_no)).where(
                MediaStudioRunEventEntity.run_id == run_id
            )
        )
        return int(value or 0) + 1

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(MediaStudioRunEventEntity).where(
                MediaStudioRunEventEntity.domain_id == domain_id,
                MediaStudioRunEventEntity.run_id == run_id,
            )
        )

class MediaStudioPromptRevisionRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: MediaStudioPromptRevisionEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def list_by_run(
        self, *, domain_id: int, run_id: UUID,
    ) -> list[MediaStudioPromptRevisionEntity]:
        rows = await self._session.scalars(
            select(MediaStudioPromptRevisionEntity)
            .where(
                MediaStudioPromptRevisionEntity.domain_id == domain_id,
                MediaStudioPromptRevisionEntity.run_id == run_id,
            )
            .order_by(MediaStudioPromptRevisionEntity.version_no)
        )
        return list(rows)

    async def get(
        self, *, prompt_revision_id: UUID,
    ) -> MediaStudioPromptRevisionEntity | None:
        return (
            await self._session.execute(
                select(MediaStudioPromptRevisionEntity).where(
                    MediaStudioPromptRevisionEntity.prompt_revision_id == prompt_revision_id
                )
            )
        ).scalar_one_or_none()

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(MediaStudioPromptRevisionEntity).where(
                MediaStudioPromptRevisionEntity.domain_id == domain_id,
                MediaStudioPromptRevisionEntity.run_id == run_id,
            )
        )

class MediaStudioMediaAssetRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row: MediaStudioMediaAssetEntity) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get(
        self, *, domain_id: int, asset_id: UUID,
    ) -> MediaStudioMediaAssetEntity | None:
        return (
            await self._session.execute(
                select(MediaStudioMediaAssetEntity).where(
                    MediaStudioMediaAssetEntity.domain_id == domain_id,
                    MediaStudioMediaAssetEntity.asset_id == asset_id,
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
    ) -> list[MediaStudioMediaAssetEntity]:
        conditions = [MediaStudioMediaAssetEntity.domain_id == domain_id]
        if run_id is not None:
            conditions.append(MediaStudioMediaAssetEntity.run_id == run_id)
        if status:
            conditions.append(MediaStudioMediaAssetEntity.status == status)
        rows = await self._session.scalars(
            select(MediaStudioMediaAssetEntity)
            .where(*conditions)
            .order_by(
                MediaStudioMediaAssetEntity.created_at.desc(),
                MediaStudioMediaAssetEntity.asset_id,
            )
            .limit(limit)
        )
        return list(rows)

    async def delete_by_run(self, *, domain_id: int, run_id: UUID) -> None:
        await self._session.execute(
            delete(MediaStudioMediaAssetEntity).where(
                MediaStudioMediaAssetEntity.domain_id == domain_id,
                MediaStudioMediaAssetEntity.run_id == run_id,
            )
        )
