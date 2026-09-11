"""智能工作台 Unit of Work。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from assistant_app.repositories import AssistantAgentRepository


class AssistantAppUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.agents: AssistantAgentRepository | None = None
        self._committed = False

    async def __aenter__(self):
        self.session = self._session_factory()
        self.agents = AssistantAgentRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("智能工作台 UoW 尚未进入事务")
        await self.session.commit()
        self._committed = True

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


def create_assistant_app_uow(session_factory: async_sessionmaker[AsyncSession]):
    return lambda: AssistantAppUnitOfWork(session_factory)
