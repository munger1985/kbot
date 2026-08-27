"""Main API 的事务边界。"""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from main_api.repositories import (
    AccessControlRepository,
    AppApiKeyRepository,
    NotificationRepository,
    PlatformDomainRepository,
)


class MainApiUnitOfWork:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self.session: AsyncSession | None = None
        self.domains: PlatformDomainRepository | None = None
        self.notifications: NotificationRepository | None = None
        self.access: AccessControlRepository | None = None
        self.app_api_keys: AppApiKeyRepository | None = None

    async def __aenter__(self) -> "MainApiUnitOfWork":
        self.session = self._session_factory()
        self.domains = PlatformDomainRepository(self.session)
        self.notifications = NotificationRepository(self.session)
        self.access = AccessControlRepository(self.session)
        self.app_api_keys = AppApiKeyRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            return
        session = self.session
        cancelled = self._contains_cancellation(exc_type, exc)
        try:
            if cancelled:
                # 流式请求取消时驱动可能已关闭连接，不能再发送 rollback。
                try:
                    await session.invalidate()
                except BaseException:
                    # 保留原始取消信号，避免清理异常污染 ASGI 日志。
                    pass
            else:
                if session.in_transaction():
                    await session.rollback()
                await session.close()
        finally:
            self.session = None
            self.domains = None
            self.notifications = None
            self.access = None
            self.app_api_keys = None

    @staticmethod
    def _contains_cancellation(exc_type, exc) -> bool:
        """识别直接取消及 ASGI TaskGroup 包装后的取消异常。"""
        if exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
            return True
        pending = [exc]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            if isinstance(current, asyncio.CancelledError):
                return True
            children = getattr(current, "exceptions", ())
            if isinstance(children, (tuple, list)):
                pending.extend(children)
        return False

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("Main API UoW 尚未进入事务上下文")
        await self.session.commit()


def create_main_api_uow(
    session_factory: async_sessionmaker[AsyncSession],
):
    """创建由 Application Service 控制的 UoW Factory。"""
    return lambda: MainApiUnitOfWork(session_factory)
