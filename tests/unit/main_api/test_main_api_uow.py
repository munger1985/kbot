"""Main API 事务清理行为测试。"""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from main_api.persistence.uow import MainApiUnitOfWork


class _CancellationGroup(BaseException):
    """模拟不同 Python/AnyIO 版本提供的取消异常组。"""

    def __init__(self, exceptions):
        self.exceptions = tuple(exceptions)


class MainApiUnitOfWorkTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_stream_invalidates_without_rollback(self):
        session = SimpleNamespace(
            in_transaction=lambda: True,
            rollback=AsyncMock(),
            invalidate=AsyncMock(),
            close=AsyncMock(),
        )
        uow = MainApiUnitOfWork(lambda: session)
        await uow.__aenter__()

        await uow.__aexit__(
            asyncio.CancelledError,
            asyncio.CancelledError(),
            None,
        )

        session.invalidate.assert_awaited_once()
        session.rollback.assert_not_awaited()
        session.close.assert_not_awaited()
        self.assertIsNone(uow.session)

    async def test_normal_read_transaction_rolls_back_and_closes(self):
        session = SimpleNamespace(
            in_transaction=lambda: True,
            rollback=AsyncMock(),
            invalidate=AsyncMock(),
            close=AsyncMock(),
        )
        uow = MainApiUnitOfWork(lambda: session)
        await uow.__aenter__()

        await uow.__aexit__(None, None, None)

        session.rollback.assert_awaited_once()
        session.close.assert_awaited_once()
        session.invalidate.assert_not_awaited()

    async def test_cancelled_stream_inside_exception_group_invalidates(self):
        session = SimpleNamespace(
            in_transaction=lambda: True,
            rollback=AsyncMock(),
            invalidate=AsyncMock(),
            close=AsyncMock(),
        )
        uow = MainApiUnitOfWork(lambda: session)
        await uow.__aenter__()
        grouped = _CancellationGroup([asyncio.CancelledError()])

        await uow.__aexit__(_CancellationGroup, grouped, None)

        session.invalidate.assert_awaited_once()
        session.rollback.assert_not_awaited()
        session.close.assert_not_awaited()
        self.assertIsNone(uow.session)


if __name__ == "__main__":
    unittest.main()
