"""Target 恢复目标和恢复演练 Repository。"""

from collections.abc import Callable
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import RecoveryDrillEntity, RecoveryProfileEntity
from aiops_agent.repositories._base import AIOpsRepository


class RecoveryRepository(AIOpsRepository):
    def __init__(self, session: AsyncSession, assert_active: Callable[[], None] | None = None) -> None:
        super().__init__(session, assert_active)

    async def add_profile(self, entity: RecoveryProfileEntity) -> RecoveryProfileEntity:
        return await self._add(entity)

    async def add_drill(self, entity: RecoveryDrillEntity) -> RecoveryDrillEntity:
        return await self._add(entity)

    async def list_profile_versions(self, *, target_id: UUID, domain_id: int, lock: bool = False) -> list[RecoveryProfileEntity]:
        self._check_active()
        statement: Select = select(RecoveryProfileEntity).where(
            RecoveryProfileEntity.target_id == target_id,
            RecoveryProfileEntity.domain_id == domain_id,
        ).order_by(RecoveryProfileEntity.version_no.desc(), RecoveryProfileEntity.created_at.desc())
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())

    async def get_active_profile(self, *, target_id: UUID, domain_id: int) -> RecoveryProfileEntity | None:
        self._check_active()
        return (await self._session.execute(
            select(RecoveryProfileEntity).where(
                RecoveryProfileEntity.target_id == target_id,
                RecoveryProfileEntity.domain_id == domain_id,
                RecoveryProfileEntity.status == "ACTIVE",
            ).order_by(RecoveryProfileEntity.version_no.desc()).limit(1)
        )).scalar_one_or_none()

    async def get_drill_scoped(self, *, drill_id: UUID, target_id: UUID, domain_id: int, lock: bool = False) -> RecoveryDrillEntity | None:
        self._check_active()
        statement: Select = select(RecoveryDrillEntity).where(
            RecoveryDrillEntity.drill_id == drill_id,
            RecoveryDrillEntity.target_id == target_id,
            RecoveryDrillEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_recent_drills(self, *, target_id: UUID, domain_id: int, limit: int = 50) -> list[RecoveryDrillEntity]:
        self._check_active()
        return list((await self._session.execute(
            select(RecoveryDrillEntity).where(
                RecoveryDrillEntity.target_id == target_id,
                RecoveryDrillEntity.domain_id == domain_id,
            ).order_by(RecoveryDrillEntity.simulated_failure_at.desc(), RecoveryDrillEntity.created_at.desc()).limit(limit)
        )).scalars())

    async def latest_attempt(self, *, target_id: UUID, domain_id: int) -> RecoveryDrillEntity | None:
        rows = await self.list_recent_drills(target_id=target_id, domain_id=domain_id, limit=1)
        return rows[0] if rows else None

    async def latest_verified_success(self, *, target_id: UUID, domain_id: int) -> RecoveryDrillEntity | None:
        self._check_active()
        return (await self._session.execute(
            select(RecoveryDrillEntity).where(
                RecoveryDrillEntity.target_id == target_id,
                RecoveryDrillEntity.domain_id == domain_id,
                RecoveryDrillEntity.status == "VERIFIED",
                RecoveryDrillEntity.result == "PASS",
            ).order_by(RecoveryDrillEntity.simulated_failure_at.desc(), RecoveryDrillEntity.created_at.desc()).limit(1)
        )).scalar_one_or_none()
