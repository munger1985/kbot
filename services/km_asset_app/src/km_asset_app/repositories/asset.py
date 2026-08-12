"""KM Asset Repository。"""

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from km_asset_app.entities import KmAssetEntity, KmAssetRevisionEntity, KmAttachmentEntity, KmJobEntity, KmSourceEntity


class KmAssetRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, row) -> None:
        self._session.add(row)
        await self._session.flush()

    async def get_source(self, *, domain_id: int, source_id: UUID, lock: bool = False):
        statement = select(KmSourceEntity).where(KmSourceEntity.domain_id == domain_id, KmSourceEntity.source_id == source_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_sources(self, *, domain_id: int):
        rows = await self._session.scalars(select(KmSourceEntity).where(KmSourceEntity.domain_id == domain_id).order_by(KmSourceEntity.updated_at.desc()))
        return list(rows)

    async def list_active_sources(self):
        rows = await self._session.scalars(select(KmSourceEntity).where(KmSourceEntity.status == "ACTIVE").order_by(KmSourceEntity.source_id))
        return list(rows)

    async def get_asset(self, *, domain_id: int, km_asset_id: UUID, lock: bool = False):
        statement = select(KmAssetEntity).where(KmAssetEntity.domain_id == domain_id, KmAssetEntity.km_asset_id == km_asset_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_asset(self, *, source_id: UUID, external_asset_id: str, lock: bool = False):
        statement = select(KmAssetEntity).where(KmAssetEntity.source_id == source_id, KmAssetEntity.external_asset_id == external_asset_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_assets(self, *, domain_id: int, source_id: UUID | None, ingestion_status: str | None, offset: int, limit: int):
        conditions = [KmAssetEntity.domain_id == domain_id]
        if source_id is not None:
            conditions.append(KmAssetEntity.source_id == source_id)
        if ingestion_status is not None:
            conditions.append(KmAssetEntity.ingestion_status == ingestion_status)
        statement = select(KmAssetEntity).where(*conditions).order_by(KmAssetEntity.synced_at.desc()).offset(offset).limit(limit)
        return list(await self._session.scalars(statement))

    async def next_revision_no(self, *, km_asset_id: UUID) -> int:
        value = await self._session.scalar(select(func.max(KmAssetRevisionEntity.revision_no)).where(KmAssetRevisionEntity.km_asset_id == km_asset_id))
        return int(value or 0) + 1

    async def list_attachments(self, *, asset_revision_id: UUID):
        rows = await self._session.scalars(select(KmAttachmentEntity).where(KmAttachmentEntity.asset_revision_id == asset_revision_id).order_by(KmAttachmentEntity.ordinal_no))
        return list(rows)

    async def find_attachment(self, *, asset_revision_id: UUID, external_document_id: str):
        statement = select(KmAttachmentEntity).where(
            KmAttachmentEntity.asset_revision_id == asset_revision_id,
            KmAttachmentEntity.external_document_id == external_document_id,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_revision(self, *, asset_revision_id: UUID):
        return (await self._session.execute(select(KmAssetRevisionEntity).where(KmAssetRevisionEntity.asset_revision_id == asset_revision_id))).scalar_one_or_none()

    async def get_job(self, *, domain_id: int, job_id: UUID):
        return (await self._session.execute(select(KmJobEntity).where(KmJobEntity.domain_id == domain_id, KmJobEntity.job_id == job_id))).scalar_one_or_none()

    async def get_job_by_id(self, *, job_id: UUID, lock: bool = False):
        statement = select(KmJobEntity).where(KmJobEntity.job_id == job_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_job_by_key(self, *, domain_id: int, idempotency_key: str):
        return (await self._session.execute(select(KmJobEntity).where(KmJobEntity.domain_id == domain_id, KmJobEntity.idempotency_key == idempotency_key))).scalar_one_or_none()

    async def list_jobs(self, *, domain_id: int, source_id: UUID | None, limit: int):
        conditions = [KmJobEntity.domain_id == domain_id]
        if source_id is not None:
            conditions.append(KmJobEntity.source_id == source_id)
        rows = await self._session.scalars(select(KmJobEntity).where(*conditions).order_by(KmJobEntity.created_at.desc()).limit(limit))
        return list(rows)

    async def claim_job(self, *, worker_id: str, lease_until: datetime):
        now = datetime.now(timezone.utc)
        eligibility = (
            or_(
                KmJobEntity.status.in_(("PENDING", "RETRY_WAIT")),
                (KmJobEntity.status == "RUNNING")
                & (KmJobEntity.lease_until < now),
            ),
            KmJobEntity.available_at <= now,
        )
        # Oracle 不允许 FETCH FIRST 与 FOR UPDATE 作用于同一查询块。
        # 先无锁读取少量有序候选主键，再逐个按主键锁定并跳过并发 Worker 已锁行。
        candidate_statement = (
            select(KmJobEntity.job_id)
            .where(
                *eligibility,
            )
            .order_by(KmJobEntity.priority.desc(), KmJobEntity.created_at)
            .limit(32)
        )
        candidate_ids = list(
            (await self._session.execute(candidate_statement)).scalars()
        )
        for job_id in candidate_ids:
            lock_statement = (
                select(KmJobEntity)
                .where(KmJobEntity.job_id == job_id, *eligibility)
                .with_for_update(skip_locked=True)
            )
            row = (
                await self._session.execute(lock_statement)
            ).scalar_one_or_none()
            if row is None:
                continue
            row.status = "RUNNING"
            row.lease_owner = worker_id
            row.lease_until = lease_until
            row.attempt_count += 1
            await self._session.flush()
            return row
        return None
