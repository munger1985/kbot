"""KM Asset App Unit of Work。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from km_asset_app.repositories import (
    KmAgentRepository,
    KmAssetRepository,
    SlackIntegrationRepository,
)
from platform_core.managed_credentials import ManagedCredentialRepository


class KmAssetUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.assets: KmAssetRepository | None = None
        self.agents: KmAgentRepository | None = None
        self.managed_credentials: ManagedCredentialRepository | None = None
        self.slack: SlackIntegrationRepository | None = None
        self._committed = False

    async def __aenter__(self):
        self.session = self._session_factory()
        self.assets = KmAssetRepository(self.session)
        self.agents = KmAgentRepository(self.session)
        self.managed_credentials = ManagedCredentialRepository(self.session)
        self.slack = SlackIntegrationRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("KM Asset UoW 尚未进入事务")
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
            self.assets = None
            self.agents = None
            self.managed_credentials = None
            self.slack = None


def create_km_asset_uow(session_factory: async_sessionmaker[AsyncSession]):
    return lambda: KmAssetUnitOfWork(session_factory)
