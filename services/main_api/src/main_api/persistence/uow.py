"""Main API 的事务边界。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from main_api.repositories import (
    PlatformDomainRepository,
    SlackIntegrationRepository,
)


class MainApiUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.domains: PlatformDomainRepository | None = None
        self.slack: SlackIntegrationRepository | None = None

    async def __aenter__(self) -> "MainApiUnitOfWork":
        self.session = self._session_factory()
        self.domains = PlatformDomainRepository(self.session)
        self.slack = SlackIntegrationRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            return
        if self.session.in_transaction():
            await self.session.rollback()
        await self.session.close()
        self.session = None
        self.domains = None
        self.slack = None

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("Main API UoW 尚未进入事务上下文")
        await self.session.commit()


def create_main_api_uow(
    session_factory: async_sessionmaker[AsyncSession],
):
    """创建由 Application Service 控制的 UoW Factory。"""
    return lambda: MainApiUnitOfWork(session_factory)
