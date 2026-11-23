"""独立数据保护配置仓储。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import or_, select, text

from aiops_agent.entities.data_protection import ProtectionPlanEntity, ProtectionRunEntity
from aiops_agent.repositories._base import AIOpsRepository


class ProtectionPlanRepository(AIOpsRepository):
    async def get(self, *, resource_id: UUID, domain_id: int, lock: bool = False):
        statement = select(ProtectionPlanEntity).where(
            ProtectionPlanEntity.resource_id == resource_id,
            ProtectionPlanEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, row: ProtectionPlanEntity) -> ProtectionPlanEntity:
        return await self._add(row)

    async def due_plan(
        self, *, operation: str, now: datetime, resource_ids: frozenset[UUID] | None = None,
    ) -> ProtectionPlanEntity | None:
        due = (ProtectionPlanEntity.next_backup_at if operation == "BACKUP"
               else ProtectionPlanEntity.next_drill_at)
        statement = select(ProtectionPlanEntity).where(due <= now).order_by(due, ProtectionPlanEntity.resource_id)
        if resource_ids is not None:
            statement = statement.where(ProtectionPlanEntity.resource_id.in_(resource_ids))
        # Oracle 不支持 FETCH FIRST 与 FOR UPDATE 同用；先找候选，再锁定并检查时点。
        candidate = (await self._session.execute(
            statement.with_only_columns(ProtectionPlanEntity.resource_id).limit(1),
        )).scalar_one_or_none()
        if candidate is None:
            return None
        return (await self._session.execute(statement.where(
            ProtectionPlanEntity.resource_id == candidate,
        ).with_for_update(skip_locked=True))).scalar_one_or_none()

    async def pending_drill(self, resource_ids: frozenset[UUID]) -> ProtectionPlanEntity | None:
        statement = select(ProtectionPlanEntity).where(
            ProtectionPlanEntity.resource_id.in_(resource_ids),
            ProtectionPlanEntity.next_drill_at.is_(None),
            ProtectionPlanEntity.approval_json.is_not(None),
        ).order_by(ProtectionPlanEntity.resource_id)
        plans = (await self._session.scalars(statement)).all()
        return next((plan for plan in plans
                     if (plan.approval_json or {}).get("DRILL_REQUEST", {}).get("status") == "PENDING"), None)

    async def add_run(self, row: ProtectionRunEntity) -> ProtectionRunEntity:
        existing = await self._locked_conflict(row, include_active=False)
        return existing if existing is not None else await self._add(row)

    async def plan_for_run(self, run: ProtectionRunEntity) -> ProtectionPlanEntity:
        return (await self._session.execute(select(ProtectionPlanEntity).where(
            ProtectionPlanEntity.resource_id == run.resource_id,
        ))).scalar_one()

    async def reserve(self, row: ProtectionRunEntity) -> bool:
        """在同一认领事务内检查触发和共用目的地，不跨外部工具执行持锁。"""
        if await self._locked_conflict(row, include_active=True) is not None:
            return False
        await self._add(row)
        return True

    async def _locked_conflict(self, row: ProtectionRunEntity, *, include_active: bool):
        # 首次占用没有可锁定的运行行；锁定现有运行表直到本次短事务提交。
        # 也检查 UNKNOWN 记录尚未释放的活跃键，不能以失败状态自动重跑。
        self._check_active()
        await self._session.execute(text("LOCK TABLE KBOT_OPS_PROTECTION_RUN IN SHARE ROW EXCLUSIVE MODE WAIT 5"))
        fire = (
            (ProtectionRunEntity.resource_id == row.resource_id)
            & (ProtectionRunEntity.operation == row.operation)
            & (ProtectionRunEntity.scheduled_at == row.scheduled_at)
        )
        if include_active and row.active_key is not None:
            fire = or_(fire, ProtectionRunEntity.active_key == row.active_key)
        return (await self._session.execute(
            select(ProtectionRunEntity).where(fire).limit(1),
        )).scalar_one_or_none()

    async def get_run(self, run_id: UUID, *, lock: bool = False) -> ProtectionRunEntity | None:
        statement = select(ProtectionRunEntity).where(ProtectionRunEntity.run_id == run_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def expiring_run(self, *, now: datetime) -> ProtectionRunEntity | None:
        candidate = (await self._session.execute(select(ProtectionRunEntity.run_id).where(
            ProtectionRunEntity.status == "RUNNING", ProtectionRunEntity.deadline_at <= now,
        ).order_by(ProtectionRunEntity.deadline_at).limit(1))).scalar_one_or_none()
        run = await self.get_run(candidate, lock=True) if candidate else None
        return run if run and run.status == "RUNNING" else None

    async def report_pending(self) -> ProtectionRunEntity | None:
        candidate = (await self._session.execute(select(ProtectionRunEntity.run_id).where(
            ProtectionRunEntity.operation == "DRILL", ProtectionRunEntity.completed_at.is_not(None),
            ProtectionRunEntity.report_id.is_(None),
        ).order_by(ProtectionRunEntity.completed_at, ProtectionRunEntity.run_id).limit(1))).scalar_one_or_none()
        run = await self.get_run(candidate, lock=True) if candidate else None
        return run if run and not run.report_id else None

    async def latest_backup(self, resource_id: UUID, *, at: datetime) -> ProtectionRunEntity | None:
        return (await self._session.execute(select(ProtectionRunEntity).where(
            ProtectionRunEntity.resource_id == resource_id, ProtectionRunEntity.operation == "BACKUP",
            ProtectionRunEntity.status == "SUCCEEDED", ProtectionRunEntity.completed_at <= at,
        ).order_by(ProtectionRunEntity.completed_at.desc(), ProtectionRunEntity.run_id.desc()).limit(1))).scalar_one_or_none()

    async def history(self, resource_id: UUID) -> list[ProtectionRunEntity]:
        return list((await self._session.scalars(select(ProtectionRunEntity).where(
            ProtectionRunEntity.resource_id == resource_id,
        ).order_by(ProtectionRunEntity.scheduled_at.desc(), ProtectionRunEntity.run_id.desc()).limit(20))).all())
