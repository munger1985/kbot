"""知识检索应用 Unit of Work。"""

from knowledge_retrieval_app.repositories.knowledge_library import KnowledgeLibraryRepository
from knowledge_retrieval_app.repositories.knowledge_queue import KnowledgeJobQueueRepository
from knowledge_retrieval_app.repositories.knowledge_ingestion import KnowledgeIngestionRepository
from knowledge_retrieval_app.repositories.knowledge_retrieval import KnowledgeRetrievalRepository
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_retrieval_app.repositories import (
    KnowledgeRetrievalAgentRepository,
    KnowledgeRetrievalResearchEventRepository,
    KnowledgeRetrievalResearchRunRepository,
    KnowledgeRetrievalXSourceRepository,
)


class KnowledgeRetrievalAppUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.knowledge_ingestion = None
        self.knowledge_retrieval = None
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.agents: KnowledgeRetrievalAgentRepository | None = None
        self.runs: KnowledgeRetrievalResearchRunRepository | None = None
        self.run_events: KnowledgeRetrievalResearchEventRepository | None = None
        self.x_sources: KnowledgeRetrievalXSourceRepository | None = None
        self._committed = False

    async def __aenter__(self):
        self._committed = False
        self.session = self._session_factory()
        self.knowledge_ingestion = KnowledgeIngestionRepository(self.session, self._require_knowledge_active)
        self.knowledge_retrieval = KnowledgeRetrievalRepository(self.session, self._require_knowledge_active)
        self.knowledge_queue = KnowledgeJobQueueRepository(self.session, self._require_knowledge_active)
        self.agents = KnowledgeRetrievalAgentRepository(self.session)
        self.runs = KnowledgeRetrievalResearchRunRepository(self.session)
        self.run_events = KnowledgeRetrievalResearchEventRepository(self.session)
        self.x_sources = KnowledgeRetrievalXSourceRepository(self.session)
        return self

    def library(self, scope):
        self._require_knowledge_active()
        return KnowledgeLibraryRepository(self.session, scope)

    def _require_knowledge_active(self):
        if self.session is None or self._committed:
            raise RuntimeError("知识摄取仓储已离开活动事务")

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
            self.knowledge_ingestion = None
            self.knowledge_retrieval = None
            self.session = None
            self.agents = None
            self.runs = None
            self.run_events = None
            self.x_sources = None


def create_knowledge_retrieval_app_uow(session_factory: async_sessionmaker[AsyncSession]):
    return lambda: KnowledgeRetrievalAppUnitOfWork(session_factory)
