"""Slack Inbox/Delivery 抢占的 Oracle 查询结构回归测试。"""

from types import SimpleNamespace
import unittest

from sqlalchemy.dialects import oracle

from km_asset_app.repositories.slack import SlackIntegrationRepository
from platform_core.identity import uuid7


class _Result:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value


class _Session:
    def __init__(self, results):
        self.results = iter(results)
        self.statements = []
        self.flush_count = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(next(self.results))

    async def flush(self):
        self.flush_count += 1


class SlackRepositoryOracleTest(unittest.IsolatedAsyncioTestCase):
    async def test_inbox_claim_locks_primary_key_without_fetch_first(self):
        first_id = uuid7()
        second_id = uuid7()
        row = SimpleNamespace(
            inbox_id=second_id,
            lease_owner=None,
            lease_until=None,
            updated_at=None,
        )
        session = _Session([[first_id, second_id], None, row])

        claimed = await SlackIntegrationRepository(session).claim_inbox(
            worker_id="slack-worker-1",
            lease_seconds=60,
        )

        self.assertIs(row, claimed)
        self.assertEqual("slack-worker-1", row.lease_owner)
        self.assertIsNotNone(row.lease_until)
        self.assertIsNotNone(row.updated_at)
        self.assertEqual(1, session.flush_count)
        self._assert_candidate_and_locks(
            session.statements,
            '"KBOT_KM_SLACK_INBOX".INBOX_ID',
        )

    async def test_delivery_claim_locks_primary_key_without_fetch_first(self):
        first_id = uuid7()
        second_id = uuid7()
        row = SimpleNamespace(
            delivery_id=second_id,
            lease_owner=None,
            lease_until=None,
            attempt_count=0,
        )
        session = _Session([[first_id, second_id], None, row])

        claimed = await SlackIntegrationRepository(session).claim_delivery(
            worker_id="slack-worker-1",
            lease_seconds=60,
        )

        self.assertIs(row, claimed)
        self.assertEqual("slack-worker-1", row.lease_owner)
        self.assertIsNotNone(row.lease_until)
        self.assertEqual(1, row.attempt_count)
        self.assertEqual(1, session.flush_count)
        self._assert_candidate_and_locks(
            session.statements,
            '"KBOT_KM_SLACK_DELIVERY".DELIVERY_ID',
        )

    def _assert_candidate_and_locks(self, statements, primary_key):
        self.assertEqual(3, len(statements))
        candidate_sql = self._oracle_sql(statements[0])
        self.assertIn("FETCH FIRST", candidate_sql)
        self.assertNotIn("FOR UPDATE", candidate_sql)
        for statement in statements[1:]:
            lock_sql = self._oracle_sql(statement)
            self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
            self.assertNotIn("FETCH FIRST", lock_sql)
            self.assertIn(primary_key, lock_sql)

    @staticmethod
    def _oracle_sql(statement) -> str:
        return str(
            statement.compile(
                dialect=oracle.dialect(),
                compile_kwargs={"literal_binds": False},
            )
        ).upper()


if __name__ == "__main__":
    unittest.main()
