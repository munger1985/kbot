"""AIOps Run 聚合、Task 租约、Artifact 和事件流 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, func, literal_column, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    OpsAlertEntity,
    OpsArtifactEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    TargetEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class OpsRunRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_run(self, entity: OpsRunEntity) -> OpsRunEntity:
        return await self._add(entity)

    async def database_now(self) -> datetime:
        """读取 Oracle Session 的 UTC 数据库时间。"""
        self._check_active()
        value = (
            await self._session.execute(
                select(literal_column("CURRENT_TIMESTAMP"))
            )
        ).scalar_one()
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    async def add_task(self, entity: OpsTaskEntity) -> OpsTaskEntity:
        return await self._add(entity)

    async def add_tasks(
        self, entities: list[OpsTaskEntity]
    ) -> list[OpsTaskEntity]:
        return await self._add_all(entities)

    async def add_artifact(
        self, entity: OpsArtifactEntity
    ) -> OpsArtifactEntity:
        return await self._add(entity)

    async def get_run_scoped(
        self,
        *,
        ops_run_id: UUID,
        app_id: int,
        domain_id: int,
        lock: bool = False,
    ) -> OpsRunEntity | None:
        self._check_active()
        statement: Select = (
            select(OpsRunEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == OpsRunEntity.target_id,
            )
            .where(
                OpsRunEntity.ops_run_id == ops_run_id,
                TargetEntity.app_id == app_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency(
        self,
        *,
        target_id: UUID,
        trigger_type: str,
        actor_id: str,
        idempotency_key: str,
    ) -> OpsRunEntity | None:
        self._check_active()
        statement = select(OpsRunEntity).where(
            OpsRunEntity.target_id == target_id,
            OpsRunEntity.trigger_type == trigger_type,
            OpsRunEntity.actor_id == actor_id,
            OpsRunEntity.idempotency_key == idempotency_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_task(
        self,
        *,
        ops_task_id: UUID,
        lock: bool = False,
    ) -> OpsTaskEntity | None:
        self._check_active()
        statement: Select = select(OpsTaskEntity).where(
            OpsTaskEntity.ops_task_id == ops_task_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_run(
        self,
        *,
        ops_run_id: UUID,
        lock: bool = False,
        skip_locked: bool = False,
    ) -> OpsRunEntity | None:
        self._check_active()
        statement: Select = select(OpsRunEntity).where(
            OpsRunEntity.ops_run_id == ops_run_id
        )
        if lock:
            statement = statement.with_for_update(skip_locked=skip_locked)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active_by_alert(
        self, *, alert_id: UUID
    ) -> OpsRunEntity | None:
        self._check_active()
        statement = (
            select(OpsRunEntity)
            .where(
                OpsRunEntity.trigger_alert_id == alert_id,
                OpsRunEntity.status.notin_(
                    (
                        "COMPLETED",
                        "DEGRADED",
                        "REJECTED",
                        "FAILED",
                        "CANCELLED",
                        "EXPIRED",
                    )
                ),
            )
            .order_by(OpsRunEntity.created_at.desc())
        )
        return (await self._session.execute(statement)).scalars().first()

    async def get_latest_by_alert_fingerprint(
        self, *, target_id: UUID, fingerprint: str
    ) -> OpsRunEntity | None:
        self._check_active()
        statement = (
            select(OpsRunEntity)
            .join(
                OpsAlertEntity,
                OpsAlertEntity.alert_id == OpsRunEntity.trigger_alert_id,
            )
            .where(
                OpsRunEntity.target_id == target_id,
                OpsAlertEntity.fingerprint == fingerprint,
            )
            .order_by(
                OpsRunEntity.created_at.desc(),
                OpsRunEntity.ops_run_id.desc(),
            )
        )
        return (await self._session.execute(statement)).scalars().first()

    async def list_tasks(
        self, *, ops_run_id: UUID, lock: bool = False
    ) -> list[OpsTaskEntity]:
        self._check_active()
        statement = (
            select(OpsTaskEntity)
            .where(OpsTaskEntity.ops_run_id == ops_run_id)
            .order_by(OpsTaskEntity.ops_task_id)
        )
        if lock:
            statement = statement.with_for_update()
        return list((await self._session.execute(statement)).scalars())

    async def list_artifacts(
        self, *, ops_run_id: UUID
    ) -> list[OpsArtifactEntity]:
        self._check_active()
        statement = (
            select(OpsArtifactEntity)
            .where(OpsArtifactEntity.ops_run_id == ops_run_id)
            .order_by(OpsArtifactEntity.artifact_key)
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_artifact_by_key(
        self,
        *,
        ops_run_id: UUID,
        artifact_key: str,
    ) -> OpsArtifactEntity | None:
        self._check_active()
        statement = select(OpsArtifactEntity).where(
            OpsArtifactEntity.ops_run_id == ops_run_id,
            OpsArtifactEntity.artifact_key == artifact_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_event_by_key(
        self, *, ops_run_id: UUID, event_key: str
    ) -> OpsRunEventEntity | None:
        self._check_active()
        statement = select(OpsRunEventEntity).where(
            OpsRunEventEntity.ops_run_id == ops_run_id,
            OpsRunEventEntity.event_key == event_key,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def latest_event_sequence(self, *, ops_run_id: UUID) -> int:
        self._check_active()
        value = (
            await self._session.execute(
                select(
                    func.coalesce(func.max(OpsRunEventEntity.sequence_no), 0)
                ).where(OpsRunEventEntity.ops_run_id == ops_run_id)
            )
        ).scalar_one()
        return int(value)

    async def transition_run(
        self,
        *,
        ops_run_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        values: dict | None = None,
    ) -> bool:
        self._check_active()
        update_values = dict(values or {})
        update_values.update(
            {
                "status": new_status,
                "row_version": OpsRunEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(OpsRunEntity)
            .where(
                OpsRunEntity.ops_run_id == ops_run_id,
                OpsRunEntity.row_version == expected_version,
                OpsRunEntity.status.in_(allowed_statuses),
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def claim_task(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> OpsTaskEntity | None:
        """候选查询不持锁，真正领取严格按 Run → Task 加锁。"""
        self._check_active()
        candidates = list(
            (
                await self._session.execute(
                    select(
                        OpsTaskEntity.ops_task_id,
                        OpsTaskEntity.ops_run_id,
                    )
                    .join(
                        OpsRunEntity,
                        OpsRunEntity.ops_run_id
                        == OpsTaskEntity.ops_run_id,
                    )
                    .where(
                        OpsTaskEntity.status == "READY",
                        OpsTaskEntity.available_at
                        <= literal_column("SYSTIMESTAMP"),
                        OpsTaskEntity.attempt_count
                        < OpsTaskEntity.max_attempts,
                        OpsRunEntity.status.not_in(
                            (
                                "COMPLETED",
                                "DEGRADED",
                                "REJECTED",
                                "FAILED",
                                "CANCELLED",
                                "EXPIRED",
                            )
                        ),
                        OpsRunEntity.cancel_requested_at.is_(None),
                        or_(
                            OpsRunEntity.deadline_at.is_(None),
                            OpsRunEntity.deadline_at
                            > literal_column("SYSTIMESTAMP"),
                        ),
                    )
                    .order_by(
                        OpsTaskEntity.available_at,
                        OpsTaskEntity.priority,
                        OpsTaskEntity.ops_task_id,
                    )
                    .limit(16)
                )
            ).all()
        )
        for task_id, run_id in candidates:
            run = await self.get_run(
                ops_run_id=run_id, lock=True, skip_locked=True
            )
            if (
                run is None
                or run.cancel_requested_at is not None
                or run.status
                in {
                    "COMPLETED",
                    "DEGRADED",
                    "REJECTED",
                    "FAILED",
                    "CANCELLED",
                    "EXPIRED",
                }
                or (
                    run.deadline_at is not None
                    and run.deadline_at <= now
                )
            ):
                continue
            entity = (
                await self._session.execute(
                    select(OpsTaskEntity)
                    .where(OpsTaskEntity.ops_task_id == task_id)
                    .with_for_update(skip_locked=True)
                )
            ).scalar_one_or_none()
            if (
                entity is None
                or entity.status != "READY"
                or entity.available_at > now
                or int(entity.attempt_count) >= int(entity.max_attempts)
            ):
                continue
            entity.status = "RUNNING"
            entity.lease_owner = lease_owner
            entity.lease_token = lease_token
            entity.lease_until = lease_until
            entity.heartbeat_at = now
            entity.started_at = entity.started_at or now
            entity.attempt_count = int(entity.attempt_count) + 1
            entity.error_code = None
            entity.error_message = None
            await self._session.flush()
            return entity
        return None

    async def lock_expired_task(
        self, *, now: datetime
    ) -> OpsTaskEntity | None:
        """供 Reconciler 按 Run → Task 锁定一个过期租约。"""
        self._check_active()
        candidates = (
            await self._session.execute(
                select(
                    OpsTaskEntity.ops_task_id,
                    OpsTaskEntity.ops_run_id,
                )
                .where(
                    OpsTaskEntity.status == "RUNNING",
                    OpsTaskEntity.lease_until.is_not(None),
                    OpsTaskEntity.lease_until
                    <= literal_column("SYSTIMESTAMP"),
                )
                .order_by(
                    OpsTaskEntity.lease_until,
                    OpsTaskEntity.ops_task_id,
                )
                .limit(16)
            )
        ).all()
        for task_id, run_id in candidates:
            run = await self.get_run(
                ops_run_id=run_id, lock=True, skip_locked=True
            )
            if run is None:
                continue
            locked_tasks = await self.list_tasks(
                ops_run_id=run_id, lock=True
            )
            task = next(
                (
                    item
                    for item in locked_tasks
                    if item.ops_task_id == task_id
                ),
                None,
            )
            if (
                task is not None
                and task.status == "RUNNING"
                and task.lease_until is not None
                and task.lease_until <= now
            ):
                return task
        return None

    async def lock_due_retry_task(
        self, *, now: datetime
    ) -> OpsTaskEntity | None:
        """按 Run → Task 锁定到期的 RETRY_WAIT Task。"""
        self._check_active()
        candidates = (
            await self._session.execute(
                select(
                    OpsTaskEntity.ops_task_id,
                    OpsTaskEntity.ops_run_id,
                )
                .where(
                    OpsTaskEntity.status == "RETRY_WAIT",
                    OpsTaskEntity.available_at
                    <= literal_column("SYSTIMESTAMP"),
                )
                .order_by(
                    OpsTaskEntity.available_at,
                    OpsTaskEntity.ops_task_id,
                )
                .limit(16)
            )
        ).all()
        for task_id, run_id in candidates:
            run = await self.get_run(
                ops_run_id=run_id, lock=True, skip_locked=True
            )
            if run is None:
                continue
            locked_tasks = await self.list_tasks(
                ops_run_id=run_id, lock=True
            )
            task = next(
                (
                    item
                    for item in locked_tasks
                    if item.ops_task_id == task_id
                ),
                None,
            )
            if (
                task is not None
                and task.status == "RETRY_WAIT"
                and task.available_at <= now
            ):
                return task
        return None

    async def lock_due_run(
        self, *, now: datetime
    ) -> OpsRunEntity | None:
        """锁定一个已到 Deadline 的非终态 Run。"""
        self._check_active()
        candidates = (
            await self._session.execute(
                select(OpsRunEntity.ops_run_id)
                .where(
                    OpsRunEntity.status.not_in(
                        (
                            "COMPLETED",
                            "DEGRADED",
                            "REJECTED",
                            "FAILED",
                            "CANCELLED",
                            "EXPIRED",
                        )
                    ),
                    OpsRunEntity.deadline_at.is_not(None),
                    OpsRunEntity.deadline_at
                    <= literal_column("SYSTIMESTAMP"),
                )
                .order_by(
                    OpsRunEntity.deadline_at,
                    OpsRunEntity.ops_run_id,
                )
                .limit(16)
            )
        ).scalars()
        for run_id in candidates:
            run = await self.get_run(
                ops_run_id=run_id, lock=True, skip_locked=True
            )
            if (
                run is not None
                and run.deadline_at is not None
                and run.deadline_at <= now
                and run.status
                not in {
                    "COMPLETED",
                    "DEGRADED",
                    "REJECTED",
                    "FAILED",
                    "CANCELLED",
                    "EXPIRED",
                }
            ):
                return run
        return None

    async def heartbeat_task(
        self,
        *,
        ops_task_id: UUID,
        lease_owner: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
        expected_version: int,
    ) -> bool:
        self._check_active()
        statement = (
            update(OpsTaskEntity)
            .where(
                OpsTaskEntity.ops_task_id == ops_task_id,
                OpsTaskEntity.status == "RUNNING",
                OpsTaskEntity.lease_owner == lease_owner,
                OpsTaskEntity.lease_token == lease_token,
                OpsTaskEntity.lease_until > now,
                OpsTaskEntity.row_version == expected_version,
            )
            .values(
                heartbeat_at=now,
                lease_until=lease_until,
                row_version=OpsTaskEntity.row_version + 1,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def finish_task(
        self,
        *,
        ops_task_id: UUID,
        lease_owner: str,
        lease_token: UUID,
        now: datetime,
        expected_version: int,
        new_status: str,
        output_artifact_id: UUID | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        self._check_active()
        statement = (
            update(OpsTaskEntity)
            .where(
                OpsTaskEntity.ops_task_id == ops_task_id,
                OpsTaskEntity.status == "RUNNING",
                OpsTaskEntity.lease_owner == lease_owner,
                OpsTaskEntity.lease_token == lease_token,
                OpsTaskEntity.lease_until > now,
                OpsTaskEntity.row_version == expected_version,
            )
            .values(
                status=new_status,
                output_artifact_id=output_artifact_id,
                error_code=error_code,
                error_message=error_message,
                completed_at=now,
                lease_owner=None,
                lease_token=None,
                lease_until=None,
                row_version=OpsTaskEntity.row_version + 1,
                updated_at=now,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def append_event(
        self,
        *,
        ops_run_id: UUID,
        event_type: str,
        visibility: str,
        payload_json: dict,
        ops_task_id: UUID | None = None,
        event_key: str | None = None,
    ) -> OpsRunEventEntity:
        """锁定 Run 后分配严格单调的 SSE 序号并追加事件。"""
        self._check_active()
        run = (
            await self._session.execute(
                select(OpsRunEntity)
                .where(OpsRunEntity.ops_run_id == ops_run_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if run is None:
            raise StateConflictError(f"Ops Run 不存在：{ops_run_id}")
        current = (
            await self._session.execute(
                select(
                    func.coalesce(func.max(OpsRunEventEntity.sequence_no), 0)
                ).where(OpsRunEventEntity.ops_run_id == ops_run_id)
            )
        ).scalar_one()
        entity = OpsRunEventEntity(
            ops_run_id=ops_run_id,
            ops_task_id=ops_task_id,
            sequence_no=int(current) + 1,
            event_type=event_type,
            event_key=event_key,
            visibility=visibility,
            payload_json=payload_json,
        )
        return await self._add(entity)

    async def list_events_after(
        self,
        *,
        ops_run_id: UUID,
        after_sequence: int,
        visibility: str | None = None,
        limit: int = 200,
    ) -> list[OpsRunEventEntity]:
        self._check_active()
        statement = select(OpsRunEventEntity).where(
            OpsRunEventEntity.ops_run_id == ops_run_id,
            OpsRunEventEntity.sequence_no > after_sequence,
        )
        if visibility is not None:
            statement = statement.where(
                OpsRunEventEntity.visibility == visibility
            )
        statement = statement.order_by(
            OpsRunEventEntity.sequence_no
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())
