"""多媒体创作工作台 Unit of Work。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from media_studio_app.repositories import (
    MediaStudioMediaAssetRepository,
    MediaStudioModelBindingRepository,
    MediaStudioPromptRevisionRepository,
    MediaStudioRunEventRepository,
    MediaStudioRunRepository,
)


class MediaStudioAppUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.bindings: MediaStudioModelBindingRepository | None = None
        self.runs: MediaStudioRunRepository | None = None
        self.run_events: MediaStudioRunEventRepository | None = None
        self.prompt_revisions: MediaStudioPromptRevisionRepository | None = None
        self.media_assets: MediaStudioMediaAssetRepository | None = None
        self._committed = False

    async def __aenter__(self):
        self.session = self._session_factory()
        self.bindings = MediaStudioModelBindingRepository(self.session)
        self.runs = MediaStudioRunRepository(self.session)
        self.run_events = MediaStudioRunEventRepository(self.session)
        self.prompt_revisions = MediaStudioPromptRevisionRepository(self.session)
        self.media_assets = MediaStudioMediaAssetRepository(self.session)
        return self

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("多媒体创作工作台 UoW 尚未进入事务")
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
            self.bindings = None
            self.runs = None
            self.run_events = None
            self.prompt_revisions = None
            self.media_assets = None


def create_media_studio_app_uow(session_factory: async_sessionmaker[AsyncSession]):
    return lambda: MediaStudioAppUnitOfWork(session_factory)
