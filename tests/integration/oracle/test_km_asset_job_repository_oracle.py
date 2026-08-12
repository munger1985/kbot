"""KM Asset 任务抢占的 Oracle SKIP LOCKED 查询结构回归测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from sqlalchemy.dialects import oracle

from km_asset_app.repositories.asset import KmAssetRepository
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


class KmAssetJobRepositoryOracleTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_claim_locks_by_primary_key_without_fetch_first(self):
        first_id = uuid7()
        second_id = uuid7()
        row = SimpleNamespace(
            job_id=second_id,
            status="PENDING",
            lease_owner=None,
            lease_until=None,
            attempt_count=0,
        )
        session = _Session([[first_id, second_id], None, row])
        lease_until = datetime.now(timezone.utc) + timedelta(minutes=2)

        claimed = await KmAssetRepository(session).claim_job(
            worker_id="worker-1",
            lease_until=lease_until,
        )

        self.assertIs(row, claimed)
        self.assertEqual("RUNNING", row.status)
        self.assertEqual("worker-1", row.lease_owner)
        self.assertEqual(lease_until, row.lease_until)
        self.assertEqual(1, row.attempt_count)
        self.assertEqual(1, session.flush_count)
        candidate_sql = self._oracle_sql(session.statements[0])
        self.assertIn("FETCH FIRST", candidate_sql)
        self.assertNotIn("FOR UPDATE", candidate_sql)
        for statement in session.statements[1:]:
            lock_sql = self._oracle_sql(statement)
            self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
            self.assertNotIn("FETCH FIRST", lock_sql)
            self.assertIn('"KBOT_KM_JOB".JOB_ID', lock_sql)

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
