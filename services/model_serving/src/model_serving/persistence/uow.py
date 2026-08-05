"""Model Serving 显式事务边界。"""

from collections.abc import Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from platform_core.notifications import NotificationOutboxRepository

from model_serving.common.model_repository import AIModelRepository


class ModelServingUnitOfWork:
    """目录 Application Service 的唯一事务入口。"""

    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self.models: AIModelRepository | None = None
        self.notification_outbox: NotificationOutboxRepository | None = None
        self._committed = False

    async def __aenter__(self) -> "ModelServingUnitOfWork":
        self._session = self._session_factory()
        self.models = AIModelRepository(self._session)
        self.notification_outbox = NotificationOutboxRepository(self._session)
        return self

    async def flush(self) -> None:
        if self._session is None:
            raise RuntimeError("ModelServingUnitOfWork 尚未进入事务")
        await self._session.flush()

    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("ModelServingUnitOfWork 尚未进入事务")
        await self._session.commit()
        self._committed = True

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._session is None:
            return
        try:
            if exc_type is not None or not self._committed:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None
            self.models = None
            self.notification_outbox = None


def create_model_serving_uow_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[], ModelServingUnitOfWork]:
    return lambda: ModelServingUnitOfWork(session_factory)
