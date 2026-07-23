"""AIOps Run 聚合、Task 租约、Artifact 和事件流 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
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

    async def list_tasks(self, *, ops_run_id: UUID) -> list[OpsTaskEntity]:
        self._check_active()
        statement = (
            select(OpsTaskEntity)
            .where(OpsTaskEntity.ops_run_id == ops_run_id)
            .order_by(OpsTaskEntity.created_at, OpsTaskEntity.ops_task_id)
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
        """锁定一个 READY Task 并写入本次唯一租约。"""
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT t.OPS_TASK_ID
                        FROM KBOT_OPS_TASK t
                        JOIN KBOT_OPS_RUN r
                          ON r.OPS_RUN_ID = t.OPS_RUN_ID
                        WHERE t.STATUS = 'READY'
                          AND t.AVAILABLE_AT <= SYSTIMESTAMP
                          AND t.ATTEMPT_COUNT < t.MAX_ATTEMPTS
                          AND r.STATUS NOT IN (
                              'COMPLETED', 'DEGRADED', 'REJECTED',
                              'FAILED', 'CANCELLED', 'EXPIRED'
                          )
                          AND r.CANCEL_REQUESTED_AT IS NULL
                          AND (
                              r.DEADLINE_AT IS NULL
                              OR r.DEADLINE_AT > SYSTIMESTAMP
                          )
                        ORDER BY
                            t.AVAILABLE_AT,
                            t.PRIORITY,
                            t.OPS_TASK_ID
                        FOR UPDATE OF t.OPS_TASK_ID SKIP LOCKED;
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
        entity = await self.get_task(ops_task_id=claimed_id)
        if entity is None:
            raise StateConflictError(f"领取后的 Ops Task 不存在：{claimed_id}")
        entity.status = "RUNNING"
        entity.lease_owner = lease_owner
        entity.lease_token = lease_token
        entity.lease_until = lease_until
        entity.heartbeat_at = now
        entity.started_at = entity.started_at or now
        entity.attempt_count = int(entity.attempt_count) + 1
        await self._session.flush()
        return entity

    async def lock_expired_task(
        self, *, now: datetime
    ) -> OpsTaskEntity | None:
        """供 Reconciler 锁定过期任务；本方法不直接重新领取。"""
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT OPS_TASK_ID
                        FROM KBOT_OPS_TASK
                        WHERE STATUS = 'RUNNING'
                          AND LEASE_UNTIL IS NOT NULL
                          AND LEASE_UNTIL <= SYSTIMESTAMP
                        ORDER BY LEASE_UNTIL, OPS_TASK_ID
                        FOR UPDATE OF OPS_TASK_ID SKIP LOCKED;
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
        return await self.get_task(ops_task_id=claimed_id)

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
