"""巡检 Plan、Fire 和版本化 Report 聚合 Repository。"""

from collections.abc import Callable, Collection
from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    InspectionFireEntity,
    InspectionPlanEntity,
    InspectionTargetEntity,
    ReportEntity,
    TargetEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class InspectionRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_plan(
        self, entity: InspectionPlanEntity
    ) -> InspectionPlanEntity:
        return await self._add(entity)

    async def add_target(
        self, entity: InspectionTargetEntity
    ) -> InspectionTargetEntity:
        return await self._add(entity)

    async def add_fire(
        self, entity: InspectionFireEntity
    ) -> InspectionFireEntity:
        return await self._add(entity)

    async def add_report(self, entity: ReportEntity) -> ReportEntity:
        return await self._add(entity)

    async def get_plan_scoped(
        self,
        *,
        inspection_plan_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> InspectionPlanEntity | None:
        self._check_active()
        statement: Select = select(InspectionPlanEntity).where(
            InspectionPlanEntity.inspection_plan_id == inspection_plan_id,
            InspectionPlanEntity.app_id == app_id,
            InspectionPlanEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_active_targets(
        self,
        *,
        inspection_plan_id: UUID,
        app_id: int,
        domain_id: int,
    ) -> list[InspectionTargetEntity]:
        self._check_active()
        statement = (
            select(InspectionTargetEntity)
            .join(
                InspectionPlanEntity,
                InspectionPlanEntity.inspection_plan_id
                == InspectionTargetEntity.inspection_plan_id,
            )
            .join(
                TargetEntity,
                TargetEntity.target_id == InspectionTargetEntity.target_id,
            )
            .where(
                InspectionTargetEntity.inspection_plan_id
                == inspection_plan_id,
                InspectionTargetEntity.status == "ACTIVE",
                InspectionPlanEntity.app_id == app_id,
                InspectionPlanEntity.domain_id == domain_id,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.status == "ACTIVE",
            )
            .order_by(InspectionTargetEntity.inspection_target_id)
        )
        return list((await self._session.execute(statement)).scalars())

    async def claim_due_plan(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> InspectionPlanEntity | None:
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT INSPECTION_PLAN_ID
                        FROM KBOT_OPS_INSPECTION_PLAN
                        WHERE STATUS = 'ACTIVE'
                          AND NEXT_RUN_AT IS NOT NULL
                          AND NEXT_RUN_AT <= SYSTIMESTAMP
                          AND (
                              LEASE_UNTIL IS NULL
                              OR LEASE_UNTIL <= SYSTIMESTAMP
                          )
                        ORDER BY NEXT_RUN_AT, INSPECTION_PLAN_ID
                        FOR UPDATE OF INSPECTION_PLAN_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={},
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(InspectionPlanEntity).where(
                    InspectionPlanEntity.inspection_plan_id == claimed_id
                )
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(
                f"领取后的巡检计划不存在：{claimed_id}"
            )
        entity.lease_owner = lease_owner
        entity.lease_token = lease_token
        entity.lease_until = lease_until
        await self._session.flush()
        return entity

    async def advance_claimed_plan(
        self,
        *,
        inspection_plan_id: UUID,
        lease_owner: str,
        lease_token: UUID,
        now: datetime,
        expected_version: int,
        scheduled_for: datetime,
        next_run_at: datetime | None,
        updated_by: str,
    ) -> bool:
        self._check_active()
        statement = (
            update(InspectionPlanEntity)
            .where(
                InspectionPlanEntity.inspection_plan_id
                == inspection_plan_id,
                InspectionPlanEntity.status == "ACTIVE",
                InspectionPlanEntity.lease_owner == lease_owner,
                InspectionPlanEntity.lease_token == lease_token,
                InspectionPlanEntity.lease_until > now,
                InspectionPlanEntity.row_version == expected_version,
            )
            .values(
                next_run_at=next_run_at,
                last_run_at=now,
                last_scheduled_for=scheduled_for,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
                updated_by=updated_by,
                row_version=InspectionPlanEntity.row_version + 1,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def get_fire(
        self,
        *,
        inspection_fire_id: UUID,
        lock: bool = False,
    ) -> InspectionFireEntity | None:
        self._check_active()
        statement: Select = select(InspectionFireEntity).where(
            InspectionFireEntity.inspection_fire_id == inspection_fire_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def transition_fire(
        self,
        *,
        inspection_fire_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        now: datetime,
        values: dict | None = None,
    ) -> bool:
        self._check_active()
        update_values = dict(values or {})
        update_values.update(
            {
                "status": new_status,
                "row_version": InspectionFireEntity.row_version + 1,
                "updated_at": now,
            }
        )
        statement = (
            update(InspectionFireEntity)
            .where(
                InspectionFireEntity.inspection_fire_id
                == inspection_fire_id,
                InspectionFireEntity.row_version == expected_version,
                InspectionFireEntity.status.in_(allowed_statuses),
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def publish_report(self, entity: ReportEntity) -> ReportEntity:
        """串行切换同一逻辑 Report 的当前版本。"""
        self._check_active()
        current = (
            await self._session.execute(
                select(ReportEntity)
                .where(
                    ReportEntity.ops_run_id == entity.ops_run_id,
                    ReportEntity.report_key == entity.report_key,
                    ReportEntity.is_current == 1,
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if current is not None:
            current.is_current = 0
        entity.is_current = 1
        return await self._add(entity)

    async def get_current_report_scoped(
        self,
        *,
        report_id: UUID,
        app_id: int,
        domain_id: int,
    ) -> ReportEntity | None:
        self._check_active()
        statement = (
            select(ReportEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == ReportEntity.target_id,
            )
            .where(
                ReportEntity.report_id == report_id,
                ReportEntity.is_current == 1,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        return (await self._session.execute(statement)).scalar_one_or_none()
