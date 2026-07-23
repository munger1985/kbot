"""Target、Agent Binding 与策略聚合的 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    PolicyEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetMonitorEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class TargetRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_target(self, entity: TargetEntity) -> TargetEntity:
        return await self._add(entity)

    async def add_binding(
        self, entity: TargetBindingEntity
    ) -> TargetBindingEntity:
        return await self._add(entity)

    async def add_monitor(
        self, entity: TargetMonitorEntity
    ) -> TargetMonitorEntity:
        return await self._add(entity)

    async def get_scoped(
        self,
        *,
        target_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> TargetEntity | None:
        self._check_active()
        statement: Select = select(TargetEntity).where(
            TargetEntity.target_id == target_id,
            TargetEntity.app_id == app_id,
            TargetEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_key(
        self,
        *,
        app_id: int,
        domain_id: int,
        target_key: str,
    ) -> TargetEntity | None:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.app_id == app_id,
            TargetEntity.domain_id == domain_id,
            TargetEntity.target_key == target_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(
        self,
        *,
        app_id: int,
        domain_id: int,
        statuses: Collection[str] | None = None,
    ) -> list[TargetEntity]:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.app_id == app_id,
            TargetEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(TargetEntity.status.in_(statuses))
        statement = statement.order_by(
            TargetEntity.display_name,
            TargetEntity.target_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_agent_binding(
        self,
        *,
        target_id: UUID,
        agent_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> TargetBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.target_id == target_id,
                TargetBindingEntity.agent_id == agent_id,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_monitors(
        self,
        *,
        target_id: UUID,
        app_id: int,
        domain_id: int,
        active_only: bool = True,
    ) -> list[TargetMonitorEntity]:
        self._check_active()
        statement = (
            select(TargetMonitorEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetMonitorEntity.target_id,
            )
            .where(
                TargetMonitorEntity.target_id == target_id,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if active_only:
            statement = statement.where(TargetMonitorEntity.status == "ACTIVE")
        statement = statement.order_by(
            TargetMonitorEntity.priority,
            TargetMonitorEntity.target_monitor_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def update_state(
        self,
        *,
        target_id: UUID,
        app_id: int,
        domain_id: int,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.row_version == expected_version,
                TargetEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                updated_by=updated_by,
                row_version=TargetEntity.row_version + 1,
                updated_at=datetime.now(UTC),
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_health(
        self,
        *,
        target_id: UUID,
        expected_health_version: int,
        health_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.health_version == expected_health_version,
            )
            .values(
                health_status=health_status,
                last_health_check_at=checked_at,
                last_error_code=last_error_code,
                health_version=TargetEntity.health_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class PolicyRepository(AIOpsRepository):
    async def add(self, entity: PolicyEntity) -> PolicyEntity:
        return await self._add(entity)

    async def get_active(
        self,
        *,
        app_id: int,
        domain_id: int,
        policy_key: str,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.app_id == app_id,
            PolicyEntity.domain_id == domain_id,
            PolicyEntity.policy_key == policy_key,
            PolicyEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_scoped(
        self,
        *,
        policy_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.policy_id == policy_id,
            PolicyEntity.app_id == app_id,
            PolicyEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def transition_status(
        self,
        *,
        policy_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
        effective_at: datetime | None = None,
        retired_at: datetime | None = None,
    ) -> bool:
        self._check_active()
        values = {
            "status": new_status,
            "updated_by": updated_by,
            "row_version": PolicyEntity.row_version + 1,
            "updated_at": datetime.now(UTC),
        }
        if effective_at is not None:
            values["effective_at"] = effective_at
        if retired_at is not None:
            values["retired_at"] = retired_at
        statement = (
            update(PolicyEntity)
            .where(
                PolicyEntity.policy_id == policy_id,
                PolicyEntity.row_version == expected_version,
                PolicyEntity.status.in_(allowed_statuses),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
