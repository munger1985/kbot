"""Unit tests for KC's explicit transaction boundary."""
import unittest
from unittest.mock import AsyncMock

from knowledge_core.persistence.uow import KnowledgeCoreUnitOfWork


class FakeSession:
    def __init__(self):
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.close = AsyncMock()


class KnowledgeCoreUnitOfWorkTest(unittest.IsolatedAsyncioTestCase):
    async def test_rolls_back_when_use_case_does_not_commit(self):
        session = FakeSession()

        async with KnowledgeCoreUnitOfWork(lambda: session):
            pass

        session.commit.assert_not_awaited()
        session.rollback.assert_awaited_once()
        session.close.assert_awaited_once()

    async def test_commits_only_when_application_service_requests_it(self):
        session = FakeSession()

        async with KnowledgeCoreUnitOfWork(lambda: session) as uow:
            await uow.commit()

        session.commit.assert_awaited_once()
        session.rollback.assert_not_awaited()
        session.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
