"""AIOps 显式、单次提交的 Unit of Work。"""

from collections.abc import Callable
from enum import StrEnum

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncSessionTransaction,
    async_sessionmaker,
)

from aiops_agent.application.errors import UnitOfWorkStateError
from aiops_agent.repositories import (
    AlertRepository,
    ChangeRepository,
    CredentialRepository,
    InboxRepository,
    InspectionRepository,
    MonitorSourceRepository,
    OpsRunRepository,
    OutboxRepository,
    PolicyRepository,
    TargetRepository,
)


class UnitOfWorkState(StrEnum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    COMMITTED = "COMMITTED"
    ROLLED_BACK = "ROLLED_BACK"
    CLOSED = "CLOSED"


class AIOpsUnitOfWork:
    """一个用例一个事务；正常退出但未提交时也会回滚。"""

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self._transaction: AsyncSessionTransaction | None = None
        self.state = UnitOfWorkState.NEW

        self.targets: TargetRepository | None = None
        self.credentials: CredentialRepository | None = None
        self.monitor_sources: MonitorSourceRepository | None = None
        self.policies: PolicyRepository | None = None
        self.alerts: AlertRepository | None = None
        self.runs: OpsRunRepository | None = None
        self.changes: ChangeRepository | None = None
        self.inspections: InspectionRepository | None = None
        self.inbox: InboxRepository | None = None
        self.outbox: OutboxRepository | None = None

    def _require_active(self) -> None:
        if self.state != UnitOfWorkState.ACTIVE:
            raise UnitOfWorkStateError(
                f"AIOpsUnitOfWork 当前不可写：state={self.state}"
            )

    async def __aenter__(self) -> "AIOpsUnitOfWork":
        if self.state != UnitOfWorkState.NEW:
            raise UnitOfWorkStateError("AIOpsUnitOfWork 实例不能重复进入")
        self.session = self._session_factory()
        self._transaction = await self.session.begin()
        self.state = UnitOfWorkState.ACTIVE
        guard = self._require_active
        self.targets = TargetRepository(self.session, guard)
        self.credentials = CredentialRepository(self.session, guard)
        self.monitor_sources = MonitorSourceRepository(self.session, guard)
        self.policies = PolicyRepository(self.session, guard)
        self.alerts = AlertRepository(self.session, guard)
        self.runs = OpsRunRepository(self.session, guard)
        self.changes = ChangeRepository(self.session, guard)
        self.inspections = InspectionRepository(self.session, guard)
        self.inbox = InboxRepository(self.session, guard)
        self.outbox = OutboxRepository(self.session, guard)
        return self

    async def commit(self) -> None:
        """提交一次后立即封闭 UoW，禁止开启隐式第二事务。"""
        self._require_active()
        assert self.session is not None
        assert self._transaction is not None
        await self.session.flush()
        await self._transaction.commit()
        self._transaction = None
        self.state = UnitOfWorkState.COMMITTED

    async def rollback(self) -> None:
        self._require_active()
        assert self.session is not None
        await self.session.rollback()
        self._transaction = None
        self.state = UnitOfWorkState.ROLLED_BACK

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            self.state = UnitOfWorkState.CLOSED
            return
        try:
            if self.state == UnitOfWorkState.ACTIVE:
                await self.session.rollback()
                self.state = UnitOfWorkState.ROLLED_BACK
        finally:
            await self.session.close()
            self._transaction = None
            self.session = None
            self.targets = None
            self.credentials = None
            self.monitor_sources = None
            self.policies = None
            self.alerts = None
            self.runs = None
            self.changes = None
            self.inspections = None
            self.inbox = None
            self.outbox = None
            self.state = UnitOfWorkState.CLOSED


def create_aiops_uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[], AIOpsUnitOfWork]:
    """创建供 API、Worker 和 Scheduler 注入的 UoW Factory。"""
    return lambda: AIOpsUnitOfWork(session_factory)
