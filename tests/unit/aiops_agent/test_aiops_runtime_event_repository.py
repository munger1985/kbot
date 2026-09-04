"""AIOps Run 事件 Repository 的幂等写入测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from aiops_agent.repositories.runtime import OpsRunRepository
from platform_core.identity import uuid7


class OpsRunRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_append_event_reuses_existing_event_key(self) -> None:
        session = AsyncMock()
        run_id = uuid7()
        existing = SimpleNamespace(sequence_no=7, event_key="run:terminal")
        session.execute.side_effect = (
            SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace()),
            SimpleNamespace(scalar_one_or_none=lambda: existing),
        )
        repository = OpsRunRepository(session)

        event = await repository.append_event(
            ops_run_id=run_id,
            event_type="run.expired",
            event_key="run:terminal",
            visibility="USER",
            payload_json={"status": "EXPIRED"},
        )

        self.assertIs(existing, event)
        self.assertEqual(2, session.execute.await_count)
        session.add.assert_not_called()
        session.flush.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
