"""巡检 Plan、Fire 和版本化 Report 聚合 Repository。"""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, and_, case, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    InspectionFireEntity,
    InspectionPlanEntity,
    InspectionReportTemplateEntity,
    InspectionReportTemplateVersionEntity,
    OpsConversationEntity,
    OpsConversationTurnEntity,
    OpsRunEntity,
    OutboxEntity,
    ReportEntity,
    ReportSourceEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


@dataclass(frozen=True)
class InspectionTurnState:
    """Scheduler 收敛巡检所需的最小 Turn 状态投影。"""

    turn_id: UUID
    status: str


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

    async def add_fire(
        self, entity: InspectionFireEntity
    ) -> InspectionFireEntity:
        return await self._add(entity)

    async def add_report(self, entity: ReportEntity) -> ReportEntity:
        return await self._add(entity)

    async def add_report_sources(
        self, entities: list[ReportSourceEntity]
    ) -> list[ReportSourceEntity]:
        return await self._add_all(entities)

    async def add_report_template(self, entity): return await self._add(entity)
    async def add_report_template_version(self, entity): return await self._add(entity)

    async def list_report_templates(self, *, domain_id: int):
        rows = await self._session.scalars(select(
            InspectionReportTemplateEntity
        ).where(InspectionReportTemplateEntity.domain_id == domain_id).order_by(
            InspectionReportTemplateEntity.updated_at.desc()
        ))
        return list(rows)

    async def get_report_template(self, *, domain_id: int, template_id: UUID, lock: bool = False):
        statement = select(InspectionReportTemplateEntity).where(
            InspectionReportTemplateEntity.domain_id == domain_id,
            InspectionReportTemplateEntity.template_id == template_id,
        )
        if lock: statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_report_template_version(self, *, template_version_id: UUID):
        return await self._session.get(
            InspectionReportTemplateVersionEntity, template_version_id
        )

    async def next_report_template_version(self, *, template_id: UUID) -> int:
        value = await self._session.scalar(select(func.coalesce(func.max(
            InspectionReportTemplateVersionEntity.version_no), 0)).where(
                InspectionReportTemplateVersionEntity.template_id == template_id
            ))
        return int(value) + 1

    async def get_plan_scoped(
        self,
        *,
        inspection_plan_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> InspectionPlanEntity | None:
        self._check_active()
        statement: Select = select(InspectionPlanEntity).where(
            InspectionPlanEntity.inspection_plan_id == inspection_plan_id,
            InspectionPlanEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def page_plans(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_updated_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[InspectionPlanEntity]:
        self._check_active()
        statement = select(InspectionPlanEntity).where(
            InspectionPlanEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(
                InspectionPlanEntity.status.in_(statuses)
            )
        if before_updated_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    InspectionPlanEntity.updated_at < before_updated_at,
                    and_(
                        InspectionPlanEntity.updated_at == before_updated_at,
                        InspectionPlanEntity.inspection_plan_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            InspectionPlanEntity.updated_at.desc(),
            InspectionPlanEntity.inspection_plan_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def update_plan(
        self,
        *,
        inspection_plan_id: UUID,
        domain_id: int,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": InspectionPlanEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(InspectionPlanEntity)
            .where(
                InspectionPlanEntity.inspection_plan_id
                == inspection_plan_id,
                InspectionPlanEntity.domain_id == domain_id,
                InspectionPlanEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

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

    async def get_fire_scoped(
        self,
        *,
        inspection_fire_id: UUID,
        domain_id: int,
    ) -> InspectionFireEntity | None:
        self._check_active()
        statement = (
            select(InspectionFireEntity)
            .join(
                InspectionPlanEntity,
                InspectionPlanEntity.inspection_plan_id
                == InspectionFireEntity.inspection_plan_id,
            )
            .where(
                InspectionFireEntity.inspection_fire_id
                == inspection_fire_id,
                InspectionPlanEntity.domain_id == domain_id,
            )
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def page_fires(
        self,
        *,
        domain_id: int,
        plan_id: UUID | None,
        statuses: Collection[str] | None,
        before_created_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[InspectionFireEntity]:
        self._check_active()
        statement = (
            select(InspectionFireEntity)
            .join(
                InspectionPlanEntity,
                InspectionPlanEntity.inspection_plan_id
                == InspectionFireEntity.inspection_plan_id,
            )
            .where(
                InspectionPlanEntity.domain_id == domain_id,
            )
        )
        if plan_id is not None:
            statement = statement.where(
                InspectionFireEntity.inspection_plan_id == plan_id
            )
        if statuses:
            statement = statement.where(
                InspectionFireEntity.status.in_(statuses)
            )
        if before_created_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    InspectionFireEntity.created_at < before_created_at,
                    and_(
                        InspectionFireEntity.created_at
                        == before_created_at,
                        InspectionFireEntity.inspection_fire_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            InspectionFireEntity.created_at.desc(),
            InspectionFireEntity.inspection_fire_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def list_open_fires(
        self, *, inspection_plan_id: UUID, lock: bool = False
    ) -> list[InspectionFireEntity]:
        self._check_active()
        statement: Select = (
            select(InspectionFireEntity)
            .where(
                InspectionFireEntity.inspection_plan_id
                == inspection_plan_id,
                InspectionFireEntity.status.in_(("QUEUED", "RUNNING")),
            )
            .order_by(
                InspectionFireEntity.scheduled_for,
                InspectionFireEntity.inspection_fire_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())

    async def find_reconcilable_fire(
        self,
    ) -> InspectionFireEntity | None:
        self._check_active()
        statement = (
            select(InspectionFireEntity)
            .where(
                InspectionFireEntity.status.in_(("RUNNING", "QUEUED"))
            )
            .order_by(
                case(
                    (InspectionFireEntity.status == "RUNNING", 0),
                    else_=1,
                ),
                InspectionFireEntity.updated_at,
                InspectionFireEntity.inspection_fire_id,
            )
            .limit(1)
        )
        return (await self._session.execute(statement)).scalars().first()

    async def list_runs_for_fire(
        self, *, inspection_fire_id: UUID
    ) -> list[OpsRunEntity]:
        self._check_active()
        statement = (
            select(OpsRunEntity)
            .where(
                OpsRunEntity.inspection_fire_id == inspection_fire_id
            )
            .order_by(OpsRunEntity.ops_run_id)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_agent_request_events_for_fire(
        self, *, inspection_fire_id: UUID
    ) -> list[OutboxEntity]:
        self._check_active()
        statement = (
            select(OutboxEntity)
            .where(
                OutboxEntity.aggregate_type == "OPS_INSPECTION_FIRE",
                OutboxEntity.aggregate_id == inspection_fire_id,
                OutboxEntity.event_type
                == "OPS_INSPECTION_AGENT_REQUESTED",
            )
            .order_by(OutboxEntity.outbox_id)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_turns_for_fire(
        self, *, inspection_fire_id: UUID
    ) -> list[InspectionTurnState]:
        """返回巡检 Fire 的最小 Turn 状态，避免加载无关 JSON 元数据。"""
        self._check_active()
        statement = (
            select(
                OpsConversationTurnEntity.turn_id,
                OpsConversationTurnEntity.status,
            )
            .join(
                OpsConversationEntity,
                OpsConversationEntity.conversation_id
                == OpsConversationTurnEntity.conversation_id,
            )
            .where(
                OpsConversationEntity.source_inspection_fire_id
                == inspection_fire_id
            )
            .order_by(OpsConversationTurnEntity.turn_id)
        )
        rows = (await self._session.execute(statement)).all()
        return [
            InspectionTurnState(
                turn_id=row.turn_id,
                status=row.status,
            )
            for row in rows
        ]

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
                TargetEntity.domain_id == domain_id,
            )
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_current_report_for_run(
        self, *, ops_run_id: UUID
    ) -> ReportEntity | None:
        """读取一个 Run 当前发布的报告版本。"""
        self._check_active()
        statement = (
            select(ReportEntity)
            .where(
                ReportEntity.ops_run_id == ops_run_id,
                ReportEntity.is_current == 1,
            )
            .order_by(ReportEntity.report_version.desc())
        )
        return (await self._session.execute(statement)).scalars().first()

    async def get_current_report_for_run_template(
        self, *, ops_run_id: UUID, template_id: str
    ) -> ReportEntity | None:
        """按 Run 和冻结模板精确读取当前报告，保证重复请求可重放。"""
        self._check_active()
        statement = (
            select(ReportEntity)
            .where(
                ReportEntity.ops_run_id == ops_run_id,
                ReportEntity.template_id == template_id,
                ReportEntity.is_current == 1,
            )
            .order_by(ReportEntity.report_version.desc())
        )
        return (await self._session.execute(statement)).scalars().first()

    async def get_report_scoped(
        self,
        *,
        report_id: UUID,
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
                TargetEntity.domain_id == domain_id,
            )
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def page_current_reports(
        self,
        *,
        domain_id: int,
        target_id: UUID | None,
        report_type: str | None,
        before_created_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[ReportEntity]:
        self._check_active()
        statement = (
            select(ReportEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == ReportEntity.target_id,
            )
            .where(
                ReportEntity.is_current == 1,
                TargetEntity.domain_id == domain_id,
            )
        )
        if target_id is not None:
            statement = statement.where(
                ReportEntity.target_id == target_id
            )
        if report_type is not None:
            statement = statement.where(
                ReportEntity.report_type == report_type
            )
        if before_created_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    ReportEntity.created_at < before_created_at,
                    and_(
                        ReportEntity.created_at == before_created_at,
                        ReportEntity.report_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            ReportEntity.created_at.desc(),
            ReportEntity.report_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def page_report_versions(
        self,
        *,
        ops_run_id: UUID,
        report_key: str,
        before_created_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[ReportEntity]:
        self._check_active()
        statement = select(ReportEntity).where(
            ReportEntity.ops_run_id == ops_run_id,
            ReportEntity.report_key == report_key,
        )
        if before_created_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    ReportEntity.created_at < before_created_at,
                    and_(
                        ReportEntity.created_at == before_created_at,
                        ReportEntity.report_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            ReportEntity.created_at.desc(),
            ReportEntity.report_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())
