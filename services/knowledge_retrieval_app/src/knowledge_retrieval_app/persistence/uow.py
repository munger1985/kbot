"""知识检索应用 Unit of Work。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_retrieval_app.repositories import KnowledgeRetrievalAgentRepository


class KnowledgeRetrievalAppUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.agents: KnowledgeRetrievalAgentRepository | None = None
        self._committed = False

    async def __aenter__(self):
        self.session = self._session_factory()
        self.agents = KnowledgeRetrievalAgentRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("知识检索应用 UoW 尚未进入事务")
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


def create_knowledge_retrieval_app_uow(
    session_factory: async_sessionmaker[AsyncSession],
):
    return lambda: KnowledgeRetrievalAppUnitOfWork(session_factory)
