"""Data Query 显式、单次提交的 Unit of Work。"""

from collections.abc import Callable
from enum import StrEnum

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, AsyncSessionTransaction, async_sessionmaker
from platform_core.notifications import NotificationOutboxRepository
from data_query.domain.errors import DataQueryPersistenceError

from data_query.repositories import (
    AgentBindingRepository,
    CredentialRepository,
    DataQueryAuditRepository,
    DataQueryHealthRepository,
    DataQueryEventRepository,
    DataQueryExecutionRepository,
    DataQueryResultRepository,
    DataQueryRunRepository,
    DataSourceRepository,
    PolicyBindingRepository,
    SchemaSnapshotRepository,
    SchemaSnapshotObjectRepository,
    SemanticModelRepository,
    SemanticModelGenerationJobRepository,
    SemanticModelVersionRepository,
    VerifiedQueryRepository,
    PlatformResourceAccessRepository,
    DataQueryModelReferenceRepository,
)


class UnitOfWorkState(StrEnum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    CLOSED = "CLOSED"


class DataQueryUnitOfWork:
    """一个 DQ 用例一个事务，Repository 不拥有独立提交权。"""

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._transaction: AsyncSessionTransaction | None = None
        self.state = UnitOfWorkState.NEW
        self.data_sources: DataSourceRepository | None = None
        self.credentials: CredentialRepository | None = None
        self.schema_snapshots: SchemaSnapshotRepository | None = None
        self.schema_snapshot_objects: SchemaSnapshotObjectRepository | None = None
        self.semantic_models: SemanticModelRepository | None = None
        self.semantic_model_generation_jobs: SemanticModelGenerationJobRepository | None = None
        self.semantic_model_versions: SemanticModelVersionRepository | None = None
        self.policy_bindings: PolicyBindingRepository | None = None
        self.agent_bindings: AgentBindingRepository | None = None
        self.verified_queries: VerifiedQueryRepository | None = None
        self.runs: DataQueryRunRepository | None = None
        self.executions: DataQueryExecutionRepository | None = None
        self.results: DataQueryResultRepository | None = None
        self.events: DataQueryEventRepository | None = None
        self.audits: DataQueryAuditRepository | None = None
        self.platform_access: PlatformResourceAccessRepository | None = None
        self.health: DataQueryHealthRepository | None = None
        self.model_references: DataQueryModelReferenceRepository | None = None
        self.notification_outbox: NotificationOutboxRepository | None = None

    def _require_active(self) -> None:
        if self.state != UnitOfWorkState.ACTIVE:
            raise RuntimeError(f"DataQueryUnitOfWork 当前不可写：state={self.state}")

    async def __aenter__(self) -> "DataQueryUnitOfWork":
        if self.state != UnitOfWorkState.NEW:
            raise RuntimeError("DataQueryUnitOfWork 实例不能重复进入")
        self._session = self._session_factory()
        self._transaction = await self._session.begin()
        self.state = UnitOfWorkState.ACTIVE
        guard = self._require_active
        self.data_sources = DataSourceRepository(self._session, guard)
        self.credentials = CredentialRepository(self._session, guard)
        self.schema_snapshots = SchemaSnapshotRepository(self._session, guard)
        self.schema_snapshot_objects = SchemaSnapshotObjectRepository(self._session, guard)
        self.semantic_models = SemanticModelRepository(self._session, guard)
        self.semantic_model_generation_jobs = SemanticModelGenerationJobRepository(self._session, guard)
        self.semantic_model_versions = SemanticModelVersionRepository(self._session, guard)
        self.policy_bindings = PolicyBindingRepository(self._session, guard)
        self.agent_bindings = AgentBindingRepository(self._session, guard)
        self.verified_queries = VerifiedQueryRepository(self._session, guard)
        self.runs = DataQueryRunRepository(self._session, guard)
        self.executions = DataQueryExecutionRepository(self._session, guard)
        self.results = DataQueryResultRepository(self._session, guard)
        self.events = DataQueryEventRepository(self._session, guard)
        self.audits = DataQueryAuditRepository(self._session, guard)
        self.platform_access = PlatformResourceAccessRepository(self._session)
        self.health = DataQueryHealthRepository(self._session, guard)
        self.model_references = DataQueryModelReferenceRepository(self._session)
        self.notification_outbox = NotificationOutboxRepository(self._session)
        return self

    async def commit(self) -> None:
        self._require_active()
        assert self._session is not None and self._transaction is not None
        await self._session.flush()
        try:
            await self._transaction.commit()
        except IntegrityError as exc:
            raise DataQueryPersistenceError("数据库完整性约束冲突") from exc
        self._transaction = None
        self.state = UnitOfWorkState.COMMITTED

    async def rollback(self) -> None:
        self._require_active()
        assert self._session is not None
        await self._session.rollback()
        self._transaction = None
        self.state = UnitOfWorkState.ROLLED_BACK

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._session is None:
            self.state = UnitOfWorkState.CLOSED
            return
        try:
            if self.state == UnitOfWorkState.ACTIVE:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._transaction = None
            self._session = None
            self.data_sources = None
            self.credentials = None
            self.schema_snapshots = None
            self.schema_snapshot_objects = None
            self.semantic_models = None
            self.semantic_model_generation_jobs = None
            self.semantic_model_versions = None
            self.policy_bindings = None
            self.agent_bindings = None
            self.verified_queries = None
            self.runs = None
            self.executions = None
            self.results = None
            self.events = None
            self.audits = None
            self.platform_access = None
            self.health = None
            self.model_references = None
            self.notification_outbox = None
            self.state = UnitOfWorkState.CLOSED


def create_data_query_uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[], DataQueryUnitOfWork]:
    return lambda: DataQueryUnitOfWork(session_factory)
