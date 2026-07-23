"""Agent Runtime 的显式提交 Unit of Work。"""

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent_runtime.repositories import (
    AgentDefinitionRepository,
    AgentArtifactRepository,
    AgentDelegationRepository,
    AgentRunEventRepository,
    AgentRunRepository,
    AgentTaskRepository,
)


class AgentRuntimeUnitOfWork:
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.agents: AgentDefinitionRepository | None = None
        self.runs: AgentRunRepository | None = None
        self.tasks: AgentTaskRepository | None = None
        self.artifacts: AgentArtifactRepository | None = None
        self.events: AgentRunEventRepository | None = None
        self.delegations: AgentDelegationRepository | None = None
        self._committed = False

    async def __aenter__(self) -> "AgentRuntimeUnitOfWork":
        self.session = self._session_factory()
        self.agents = AgentDefinitionRepository(self.session)
        self.runs = AgentRunRepository(self.session)
        self.tasks = AgentTaskRepository(self.session)
        self.artifacts = AgentArtifactRepository(self.session)
        self.events = AgentRunEventRepository(self.session)
        self.delegations = AgentDelegationRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("AgentRuntimeUnitOfWork 尚未进入事务")
        await self.session.commit()
        self._committed = True

    async def rollback(self) -> None:
        if self.session is not None:
            await self.session.rollback()

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            return
        try:
            if exc_type is not None or not self._committed:
                await self.session.rollback()
        finally:
            await self.session.close()
            self.session = None
            self.agents = None
            self.runs = None
            self.tasks = None
            self.artifacts = None
            self.events = None
            self.delegations = None


def create_agent_runtime_uow(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[], AgentRuntimeUnitOfWork]:
    """返回供 Application Service 注入的 UoW Factory。"""
    return lambda: AgentRuntimeUnitOfWork(session_factory)
