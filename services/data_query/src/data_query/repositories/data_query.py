"""Repository implementations for Data Query aggregates.

Repositories only use the session supplied by ``DataQueryUnitOfWork``.  They
never commit independently, which keeps an idempotency receipt, its first
event, and its audit entry in the same transaction.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from data_query.entities.data_query import (
    AgentBindingEntity,
    CredentialEntity,
    DataQueryAuditEntity,
    DataQueryEventEntity,
    DataQueryExecutionEntity,
    DataQueryResultEntity,
    DataQueryRunEntity,
    DataSourceEntity,
    PolicyBindingEntity,
    SchemaSnapshotEntity,
    SchemaSnapshotObjectEntity,
    SemanticModelEntity,
    SemanticModelGenerationJobEntity,
    SemanticModelVersionEntity,
    VerifiedQueryEntity,
)


class DataQueryRepository:
    """Base class that guards writes against an inactive Unit of Work."""

    def __init__(self, session: AsyncSession, assert_active: Callable[[], None]):
        self._session = session
        self._assert_active = assert_active

    async def _add(self, entity):
        self._assert_active()
        self._session.add(entity)
        await self._session.flush()
        return entity


class DataQueryHealthRepository(DataQueryRepository):
    async def is_ready(self) -> bool:
        result = await self._session.execute(text("SELECT 1 FROM dual"))
        return int(result.scalar_one()) == 1


class DataSourceRepository(DataQueryRepository):
    async def get_by_id(self, *, data_source_id: UUID, lock: bool = False) -> DataSourceEntity | None:
        statement: Select = select(DataSourceEntity).where(DataSourceEntity.data_source_id == data_source_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: DataSourceEntity) -> DataSourceEntity:
        return await self._add(entity)

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[DataSourceEntity]:
        statement = select(DataSourceEntity).where(DataSourceEntity.domain_id == domain_id)
        if after_id is not None:
            statement = statement.where(DataSourceEntity.data_source_id > after_id)
        statement = statement.order_by(DataSourceEntity.data_source_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())


class CredentialRepository(DataQueryRepository):
    """Data Query 专用凭据 Repository。"""

    async def add(self, entity: CredentialEntity) -> CredentialEntity:
        return await self._add(entity)

    async def get_scoped(
        self, *, credential_id: UUID, domain_id: int,
        data_source_id: UUID, active_only: bool = False,
        lock: bool = False,
    ) -> CredentialEntity | None:
        statement: Select = select(CredentialEntity).where(
            CredentialEntity.credential_id == credential_id,
            CredentialEntity.domain_id == domain_id,
            CredentialEntity.data_source_id == data_source_id,
        )
        if active_only:
            statement = statement.where(CredentialEntity.status == "ACTIVE")
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def revoke(self, entity: CredentialEntity, *, actor_id: str) -> None:
        self._assert_active()
        entity.status = "REVOKED"
        entity.updated_by = actor_id
        await self._session.flush()


class SchemaSnapshotRepository(DataQueryRepository):
    async def get_by_id(self, *, schema_snapshot_id: UUID) -> SchemaSnapshotEntity | None:
        return await self._session.get(SchemaSnapshotEntity, schema_snapshot_id)

    async def get_ready_for_source(self, *, data_source_id: UUID) -> SchemaSnapshotEntity | None:
        statement = select(SchemaSnapshotEntity).where(
            SchemaSnapshotEntity.data_source_id == data_source_id,
            SchemaSnapshotEntity.status == "READY",
        ).order_by(SchemaSnapshotEntity.completed_at.desc(), SchemaSnapshotEntity.schema_snapshot_id.desc()).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_source(self, *, data_source_id: UUID) -> list[SchemaSnapshotEntity]:
        statement = select(SchemaSnapshotEntity).where(
            SchemaSnapshotEntity.data_source_id == data_source_id
        ).order_by(SchemaSnapshotEntity.created_at.desc(), SchemaSnapshotEntity.schema_snapshot_id.desc())
        return list((await self._session.execute(statement)).scalars())

    async def get_by_source_hash(
        self, *, data_source_id: UUID, snapshot_hash: str, lock: bool = False
    ) -> SchemaSnapshotEntity | None:
        statement: Select = select(SchemaSnapshotEntity).where(
            SchemaSnapshotEntity.data_source_id == data_source_id,
            SchemaSnapshotEntity.snapshot_hash == snapshot_hash,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def supersede_previous(
        self, *, data_source_id: UUID, current_snapshot_id: UUID
    ) -> int:
        """新快照就绪后保留历史内容并关闭旧运行态快照。"""
        self._assert_active()
        result = await self._session.execute(
            update(SchemaSnapshotEntity)
            .where(
                SchemaSnapshotEntity.data_source_id == data_source_id,
                SchemaSnapshotEntity.schema_snapshot_id != current_snapshot_id,
                SchemaSnapshotEntity.status.in_(("READY", "PARTIAL_READY")),
            )
            .values(status="SUPERSEDED")
        )
        return int(result.rowcount or 0)

    async def add(self, entity: SchemaSnapshotEntity) -> SchemaSnapshotEntity:
        return await self._add(entity)

    async def claim_next_requested(self) -> SchemaSnapshotEntity | None:
        """多 Worker 安全领取一个待采集快照。"""
        self._assert_active()
        statement = (
            select(SchemaSnapshotEntity)
            .where(
                SchemaSnapshotEntity.status == "REQUESTED",
                text("ROWNUM = 1"),
            )
            .order_by(SchemaSnapshotEntity.created_at, SchemaSnapshotEntity.schema_snapshot_id)
            .with_for_update(skip_locked=True)
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is not None:
            row.status = "DISCOVERING"
        return row

    async def requeue_stale_discoveries(self, *, stale_before: datetime) -> int:
        """恢复进程中断后未进入终态的发现任务。"""
        self._assert_active()
        result = await self._session.execute(
            update(SchemaSnapshotEntity)
            .where(
                SchemaSnapshotEntity.status == "DISCOVERING",
                SchemaSnapshotEntity.updated_at < stale_before,
            )
            .values(status="REQUESTED")
        )
        return int(result.rowcount or 0)


class SchemaSnapshotObjectRepository(DataQueryRepository):
    async def add(self, entity: SchemaSnapshotObjectEntity) -> SchemaSnapshotObjectEntity:
        return await self._add(entity)

    async def add_all(self, entities: list[SchemaSnapshotObjectEntity]) -> None:
        self._assert_active()
        self._session.add_all(entities)
        await self._session.flush()

    async def get_by_id(
        self, *, schema_snapshot_object_id: UUID, lock: bool = False,
    ) -> SchemaSnapshotObjectEntity | None:
        statement: Select = select(SchemaSnapshotObjectEntity).where(
            SchemaSnapshotObjectEntity.schema_snapshot_object_id == schema_snapshot_object_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_snapshot(self, *, schema_snapshot_id: UUID) -> list[SchemaSnapshotObjectEntity]:
        statement = select(SchemaSnapshotObjectEntity).where(
            SchemaSnapshotObjectEntity.schema_snapshot_id == schema_snapshot_id
        ).order_by(SchemaSnapshotObjectEntity.schema_name, SchemaSnapshotObjectEntity.object_name)
        return list((await self._session.execute(statement)).scalars())

    async def claim_next_selected(self) -> SchemaSnapshotObjectEntity | None:
        self._assert_active()
        statement = (
            select(SchemaSnapshotObjectEntity)
            .where(
                SchemaSnapshotObjectEntity.status == "QUEUED",
                SchemaSnapshotObjectEntity.attempt_count < 3,
                text("ROWNUM = 1"),
            )
            .order_by(SchemaSnapshotObjectEntity.updated_at, SchemaSnapshotObjectEntity.schema_snapshot_object_id)
            .with_for_update(skip_locked=True)
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is not None:
            row.status = "CAPTURING"
            row.attempt_count += 1
        return row

    async def requeue_stale_captures(self, *, stale_before: datetime) -> int:
        """恢复进程中断后未完成的单对象采集。"""
        self._assert_active()
        result = await self._session.execute(
            update(SchemaSnapshotObjectEntity)
            .where(
                SchemaSnapshotObjectEntity.status == "CAPTURING",
                SchemaSnapshotObjectEntity.updated_at < stale_before,
            )
            .values(status="QUEUED", started_at=None)
        )
        return int(result.rowcount or 0)


class SemanticModelRepository(DataQueryRepository):
    async def get_by_id(self, *, semantic_model_id: UUID, lock: bool = False) -> SemanticModelEntity | None:
        statement: Select = select(SemanticModelEntity).where(SemanticModelEntity.semantic_model_id == semantic_model_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: SemanticModelEntity) -> SemanticModelEntity:
        return await self._add(entity)

    async def delete(self, entity: SemanticModelEntity) -> None:
        self._assert_active()
        await self._session.delete(entity)

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[SemanticModelEntity]:
        statement = select(SemanticModelEntity).where(SemanticModelEntity.domain_id == domain_id)
        if after_id is not None:
            statement = statement.where(SemanticModelEntity.semantic_model_id > after_id)
        statement = statement.order_by(SemanticModelEntity.semantic_model_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def search_by_ids(
        self, *, domain_id: int, semantic_model_ids: tuple[UUID, ...],
        query: str | None, publication_status: str | None,
        after_id: UUID | None, limit: int,
    ) -> list[SemanticModelEntity]:
        statement = select(SemanticModelEntity).where(
            SemanticModelEntity.domain_id == domain_id,
            SemanticModelEntity.semantic_model_id.in_(semantic_model_ids),
        )
        if query:
            statement = statement.where(SemanticModelEntity.display_name.ilike(f"%{query}%"))
        if publication_status == "PUBLISHED":
            statement = statement.where(SemanticModelEntity.active_version.is_not(None))
        elif publication_status == "UNPUBLISHED":
            statement = statement.where(SemanticModelEntity.active_version.is_(None))
        if after_id is not None:
            statement = statement.where(SemanticModelEntity.semantic_model_id > after_id)
        statement = statement.order_by(SemanticModelEntity.semantic_model_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())


class SemanticModelGenerationJobRepository(DataQueryRepository):
    async def add(self, entity: SemanticModelGenerationJobEntity) -> SemanticModelGenerationJobEntity:
        return await self._add(entity)

    async def get_by_id(self, *, generation_job_id: UUID, lock: bool = False) -> SemanticModelGenerationJobEntity | None:
        statement: Select = select(SemanticModelGenerationJobEntity).where(
            SemanticModelGenerationJobEntity.generation_job_id == generation_job_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def claim_next(
        self,
        *,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
    ) -> SemanticModelGenerationJobEntity | None:
        self._assert_active()
        statement = (
            select(SemanticModelGenerationJobEntity)
            .where(
                (SemanticModelGenerationJobEntity.status == "QUEUED")
                | (
                    (SemanticModelGenerationJobEntity.status == "RUNNING")
                    & (SemanticModelGenerationJobEntity.lease_until < now)
                    & (SemanticModelGenerationJobEntity.attempt_count < 3)
                ),
                text("ROWNUM = 1"),
            )
            .order_by(
                SemanticModelGenerationJobEntity.lease_until.nullsfirst(),
                SemanticModelGenerationJobEntity.created_at,
                SemanticModelGenerationJobEntity.generation_job_id,
            )
            .with_for_update(skip_locked=True)
        )
        row = (await self._session.execute(statement)).scalar_one_or_none()
        if row is not None:
            row.status = "RUNNING"
            row.attempt_count = int(row.attempt_count) + 1
            row.lease_owner = worker_id
            row.lease_token = lease_token
            row.lease_until = lease_until
            if row.started_at is None:
                row.started_at = now
            await self._session.flush()
        return row

    async def heartbeat(
        self,
        *,
        generation_job_id: UUID,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
    ) -> bool:
        self._assert_active()
        result = await self._session.execute(
            update(SemanticModelGenerationJobEntity)
            .where(
                SemanticModelGenerationJobEntity.generation_job_id == generation_job_id,
                SemanticModelGenerationJobEntity.status == "RUNNING",
                SemanticModelGenerationJobEntity.lease_owner == worker_id,
                SemanticModelGenerationJobEntity.lease_token == lease_token,
                SemanticModelGenerationJobEntity.lease_until > now,
            )
            .values(lease_until=lease_until)
        )
        return bool(result.rowcount)


class SemanticModelVersionRepository(DataQueryRepository):
    async def next_version_no(self, *, semantic_model_id: UUID) -> int:
        self._assert_active()
        statement = select(func.coalesce(func.max(SemanticModelVersionEntity.version_no), 0)).where(
            SemanticModelVersionEntity.semantic_model_id == semantic_model_id
        )
        return int((await self._session.execute(statement)).scalar_one()) + 1

    async def get_by_id(self, *, semantic_model_version_id: UUID, lock: bool = False) -> SemanticModelVersionEntity | None:
        statement: Select = select(SemanticModelVersionEntity).where(
            SemanticModelVersionEntity.semantic_model_version_id == semantic_model_version_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_model_version(
        self, *, semantic_model_id: UUID, version_no: int, lock: bool = False
    ) -> SemanticModelVersionEntity | None:
        statement: Select = select(SemanticModelVersionEntity).where(
            SemanticModelVersionEntity.semantic_model_id == semantic_model_id,
            SemanticModelVersionEntity.version_no == version_no,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active(
        self, *, semantic_model_id: UUID, lock: bool = False
    ) -> SemanticModelVersionEntity | None:
        statement = select(SemanticModelVersionEntity).where(
            SemanticModelVersionEntity.semantic_model_id == semantic_model_id,
            SemanticModelVersionEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def retire_active_except(
        self, *, semantic_model_id: UUID, keep_version_id: UUID
    ) -> int:
        """先落库旧 ACTIVE 的退役状态，避免条件唯一索引更新顺序竞争。"""
        self._assert_active()
        result = await self._session.execute(
            update(SemanticModelVersionEntity)
            .where(
                SemanticModelVersionEntity.semantic_model_id == semantic_model_id,
                SemanticModelVersionEntity.status == "ACTIVE",
                SemanticModelVersionEntity.semantic_model_version_id != keep_version_id,
            )
            .values(status="RETIRED")
        )
        return int(result.rowcount or 0)

    async def add(self, entity: SemanticModelVersionEntity) -> SemanticModelVersionEntity:
        return await self._add(entity)

    async def list_by_model(self, *, semantic_model_id: UUID) -> list[SemanticModelVersionEntity]:
        statement = select(SemanticModelVersionEntity).where(
            SemanticModelVersionEntity.semantic_model_id == semantic_model_id
        ).order_by(SemanticModelVersionEntity.version_no.desc())
        return list((await self._session.execute(statement)).scalars())

    async def delete_by_model(self, *, semantic_model_id: UUID) -> None:
        self._assert_active()
        await self._session.execute(
            delete(SemanticModelVersionEntity).where(
                SemanticModelVersionEntity.semantic_model_id == semantic_model_id,
            )
        )


class PolicyBindingRepository(DataQueryRepository):
    async def get_by_id(self, *, policy_binding_id: UUID, lock: bool = False) -> PolicyBindingEntity | None:
        statement = select(PolicyBindingEntity).where(PolicyBindingEntity.policy_binding_id == policy_binding_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: PolicyBindingEntity) -> PolicyBindingEntity:
        return await self._add(entity)

    async def references_model(self, *, domain_id: int, semantic_model_id: UUID) -> bool:
        rows = await self.list_by_domain(
            domain_id=domain_id, after_id=None, limit=10_000
        )
        return any(
            str(semantic_model_id) in (row.semantic_model_ids_json or [])
            for row in rows
        )

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[PolicyBindingEntity]:
        statement = select(PolicyBindingEntity).where(PolicyBindingEntity.domain_id == domain_id)
        if after_id is not None:
            statement = statement.where(PolicyBindingEntity.policy_binding_id > after_id)
        statement = statement.order_by(PolicyBindingEntity.policy_binding_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())


class AgentBindingRepository(DataQueryRepository):
    async def list_active_for_agent(
        self, *, domain_id: int, agent_id: UUID
    ) -> list[AgentBindingEntity]:
        statement = select(AgentBindingEntity).where(
            AgentBindingEntity.domain_id == domain_id,
            AgentBindingEntity.agent_id == agent_id,
            AgentBindingEntity.status == "ACTIVE",
        ).order_by(AgentBindingEntity.semantic_model_id, AgentBindingEntity.agent_binding_id)
        return list((await self._session.execute(statement)).scalars())

    async def get_active(self, *, domain_id: int, agent_id: UUID, semantic_model_id: UUID) -> AgentBindingEntity | None:
        statement = select(AgentBindingEntity).where(
            AgentBindingEntity.domain_id == domain_id,
            AgentBindingEntity.agent_id == agent_id,
            AgentBindingEntity.semantic_model_id == semantic_model_id,
            AgentBindingEntity.status == "ACTIVE",
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: AgentBindingEntity) -> AgentBindingEntity:
        return await self._add(entity)

    async def references_model(self, *, domain_id: int, semantic_model_id: UUID) -> bool:
        statement = select(AgentBindingEntity.agent_binding_id).where(
            AgentBindingEntity.domain_id == domain_id,
            AgentBindingEntity.semantic_model_id == semantic_model_id,
        ).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none() is not None

    async def get_by_id(self, *, agent_binding_id: UUID, lock: bool = False) -> AgentBindingEntity | None:
        statement = select(AgentBindingEntity).where(AgentBindingEntity.agent_binding_id == agent_binding_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[AgentBindingEntity]:
        statement = select(AgentBindingEntity).where(AgentBindingEntity.domain_id == domain_id)
        if after_id is not None:
            statement = statement.where(AgentBindingEntity.agent_binding_id > after_id)
        statement = statement.order_by(AgentBindingEntity.agent_binding_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def list_active(self, *, domain_id: int, agent_id: UUID, semantic_model_id: UUID) -> list[AgentBindingEntity]:
        statement = select(AgentBindingEntity).where(
            AgentBindingEntity.domain_id == domain_id,
            AgentBindingEntity.agent_id == agent_id,
            AgentBindingEntity.semantic_model_id == semantic_model_id,
            AgentBindingEntity.status == "ACTIVE",
        ).order_by(AgentBindingEntity.agent_binding_id)
        return list((await self._session.execute(statement)).scalars())


class VerifiedQueryRepository(DataQueryRepository):
    async def get_by_question_hash(self, *, semantic_model_version_id: UUID, question_hash: str) -> VerifiedQueryEntity | None:
        statement = select(VerifiedQueryEntity).where(
            VerifiedQueryEntity.semantic_model_version_id == semantic_model_version_id,
            VerifiedQueryEntity.question_hash == question_hash,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: VerifiedQueryEntity) -> VerifiedQueryEntity:
        return await self._add(entity)

    async def references_model(self, *, semantic_model_id: UUID) -> bool:
        statement = select(VerifiedQueryEntity.verified_query_id).join(
            SemanticModelVersionEntity,
            SemanticModelVersionEntity.semantic_model_version_id
            == VerifiedQueryEntity.semantic_model_version_id,
        ).where(
            SemanticModelVersionEntity.semantic_model_id == semantic_model_id,
        ).limit(1)
        return (await self._session.execute(statement)).scalar_one_or_none() is not None

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[VerifiedQueryEntity]:
        statement = select(VerifiedQueryEntity).join(
            SemanticModelVersionEntity,
            SemanticModelVersionEntity.semantic_model_version_id
            == VerifiedQueryEntity.semantic_model_version_id,
        ).join(
            SemanticModelEntity,
            SemanticModelEntity.semantic_model_id
            == SemanticModelVersionEntity.semantic_model_id,
        ).where(SemanticModelEntity.domain_id == domain_id)
        if after_id is not None:
            statement = statement.where(VerifiedQueryEntity.verified_query_id > after_id)
        statement = statement.order_by(VerifiedQueryEntity.verified_query_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())


class DataQueryRunRepository(DataQueryRepository):
    async def get_by_id(self, *, data_query_run_id: UUID, lock: bool = False) -> DataQueryRunEntity | None:
        statement: Select = select(DataQueryRunEntity).where(DataQueryRunEntity.data_query_run_id == data_query_run_id)
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_idempotency_key(self, *, domain_id: int, actor_id: str, idempotency_key: str, lock: bool = False) -> DataQueryRunEntity | None:
        statement: Select = select(DataQueryRunEntity).where(
            DataQueryRunEntity.domain_id == domain_id,
            DataQueryRunEntity.actor_id == actor_id,
            DataQueryRunEntity.idempotency_key == idempotency_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: DataQueryRunEntity) -> DataQueryRunEntity:
        return await self._add(entity)

    async def count_inflight(
        self, *, domain_id: int, agent_id: UUID
    ) -> int:
        statement = select(func.count()).select_from(DataQueryRunEntity).where(
            DataQueryRunEntity.domain_id == domain_id,
            DataQueryRunEntity.agent_id == agent_id,
            DataQueryRunEntity.status.in_(("QUEUED", "EXECUTING", "CANCEL_PENDING")),
        )
        return int((await self._session.execute(statement)).scalar_one())


class DataQueryExecutionRepository(DataQueryRepository):
    async def get_by_id(
        self, *, data_query_execution_id: UUID, lock: bool = False
    ) -> DataQueryExecutionEntity | None:
        statement: Select = select(DataQueryExecutionEntity).where(
            DataQueryExecutionEntity.data_query_execution_id == data_query_execution_id
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_by_run_attempt(self, *, data_query_run_id: UUID, attempt_no: int, lock: bool = False) -> DataQueryExecutionEntity | None:
        statement: Select = select(DataQueryExecutionEntity).where(
            DataQueryExecutionEntity.data_query_run_id == data_query_run_id,
            DataQueryExecutionEntity.attempt_no == attempt_no,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def claim_next(
        self,
        *,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
    ) -> DataQueryExecutionEntity | None:
        """领取并立刻写 Lease；事务提交后其他 Worker 不会重复执行。"""
        self._assert_active()
        statement: Select = select(DataQueryExecutionEntity).where(
            (DataQueryExecutionEntity.status == "QUEUED")
            | ((DataQueryExecutionEntity.status == "EXECUTING") & (DataQueryExecutionEntity.lease_until < now)),
            text("ROWNUM = 1"),
        ).order_by(DataQueryExecutionEntity.lease_until.nullsfirst(), DataQueryExecutionEntity.data_query_execution_id).with_for_update(skip_locked=True)
        execution = (await self._session.execute(statement)).scalar_one_or_none()
        if execution is None:
            return None
        execution.status = "EXECUTING"
        execution.lease_owner = worker_id
        execution.lease_token = lease_token
        execution.lease_until = lease_until
        execution.heartbeat_at = now
        if execution.started_at is None:
            execution.started_at = now
        await self._session.flush()
        return execution

    async def heartbeat(
        self,
        *,
        data_query_execution_id: UUID,
        worker_id: str,
        lease_token: UUID,
        now: datetime,
        lease_until: datetime,
    ) -> bool:
        """仅允许当前未过期租约的所有者续租。"""
        self._assert_active()
        result = await self._session.execute(
            update(DataQueryExecutionEntity)
            .where(
                DataQueryExecutionEntity.data_query_execution_id == data_query_execution_id,
                DataQueryExecutionEntity.status == "EXECUTING",
                DataQueryExecutionEntity.lease_owner == worker_id,
                DataQueryExecutionEntity.lease_token == lease_token,
                DataQueryExecutionEntity.lease_until > now,
            )
            .values(lease_until=lease_until, heartbeat_at=now)
        )
        return bool(result.rowcount)

    async def add(self, entity: DataQueryExecutionEntity) -> DataQueryExecutionEntity:
        return await self._add(entity)


class DataQueryResultRepository(DataQueryRepository):
    async def get_available_by_run_id(
        self, *, data_query_run_id: UUID, now: datetime,
    ) -> DataQueryResultEntity | None:
        statement = select(DataQueryResultEntity).where(
            DataQueryResultEntity.data_query_run_id == data_query_run_id,
            DataQueryResultEntity.purged_at.is_(None),
            DataQueryResultEntity.available_until > now,
        )
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add(self, entity: DataQueryResultEntity) -> DataQueryResultEntity:
        return await self._add(entity)

    async def purge_expired(self, *, now: datetime, limit: int) -> int:
        self._assert_active()
        statement = (
            select(DataQueryResultEntity)
            .where(
                DataQueryResultEntity.purged_at.is_(None),
                DataQueryResultEntity.available_until <= now,
                DataQueryResultEntity.storage_uri.is_(None),
                text("ROWNUM <= :purge_limit"),
            )
            .order_by(DataQueryResultEntity.available_until, DataQueryResultEntity.data_query_result_id)
            .with_for_update(skip_locked=True)
            .params(purge_limit=limit)
        )
        rows = list((await self._session.execute(statement)).scalars())
        for row in rows:
            row.columns_json = []
            row.preview_rows_json = []
            row.byte_size = 0
            row.purged_at = now
        return len(rows)


class DataQueryEventRepository(DataQueryRepository):
    async def next_sequence_no(self, *, data_query_run_id: UUID) -> int:
        self._assert_active()
        # Lock the aggregate row first: PostgreSQL does not permit ``FOR
        # UPDATE`` directly on an aggregate query, and the Run row is the
        # serialization point for its monotonically increasing event stream.
        run_statement = select(DataQueryRunEntity.data_query_run_id).where(
            DataQueryRunEntity.data_query_run_id == data_query_run_id
        ).with_for_update()
        if (await self._session.execute(run_statement)).scalar_one_or_none() is None:
            raise LookupError(f"DataQueryRun 不存在：{data_query_run_id}")
        statement = select(func.coalesce(func.max(DataQueryEventEntity.sequence_no), 0)).where(
            DataQueryEventEntity.data_query_run_id == data_query_run_id
        )
        return int((await self._session.execute(statement)).scalar_one()) + 1

    async def append(self, entity: DataQueryEventEntity) -> DataQueryEventEntity:
        return await self._add(entity)


class DataQueryAuditRepository(DataQueryRepository):
    async def latest_for_run(self, *, data_query_run_id: UUID) -> DataQueryAuditEntity | None:
        statement = select(DataQueryAuditEntity).where(
            DataQueryAuditEntity.data_query_run_id == data_query_run_id
        ).order_by(DataQueryAuditEntity.created_at.desc(), DataQueryAuditEntity.audit_id.desc()).limit(1).with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def append(self, entity: DataQueryAuditEntity) -> DataQueryAuditEntity:
        return await self._add(entity)

    async def list_by_domain(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> list[DataQueryAuditEntity]:
        statement = select(DataQueryAuditEntity).where(
            DataQueryAuditEntity.domain_id == domain_id
        )
        if after_id is not None:
            statement = statement.where(DataQueryAuditEntity.audit_id > after_id)
        statement = statement.order_by(DataQueryAuditEntity.audit_id).limit(limit)
        return list((await self._session.execute(statement)).scalars())
