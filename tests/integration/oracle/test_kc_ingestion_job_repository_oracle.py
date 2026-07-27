"""KC 入库任务的 Oracle SKIP LOCKED 查询结构回归测试。"""

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from sqlalchemy.dialects import oracle

from knowledge_core.repositories.ingestion_repo import (
    IngestionJobRepository,
)
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

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(next(self.results))


class KcIngestionJobRepositoryOracleTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_all_job_types_lock_by_primary_key_without_row_limit(
        self,
    ):
        calls = (
            ("claim_candidates", {}),
            (
                "claim_candidates_by_types",
                {
                    "job_types": (
                        "COLLECTION_PURGE",
                        "INDEX",
                        "PROFILE",
                    )
                },
            ),
        )
        now = datetime.now(timezone.utc)

        for method_name, extra in calls:
            with self.subTest(method=method_name):
                first_id = uuid7()
                second_id = uuid7()
                job = SimpleNamespace(ingestion_job_id=second_id)
                session = _Session([[first_id, second_id], None, job])
                repository = IngestionJobRepository(session)

                claimed = await getattr(repository, method_name)(
                    now=now,
                    limit=1,
                    **extra,
                )

                self.assertEqual([job], claimed)
                self.assertEqual(3, len(session.statements))
                candidate_sql = self._oracle_sql(session.statements[0])
                first_lock_sql = self._oracle_sql(session.statements[1])
                second_lock_sql = self._oracle_sql(session.statements[2])
                self.assertIn("FETCH FIRST", candidate_sql)
                self.assertNotIn("FOR UPDATE", candidate_sql)
                self.assertIn("LEASE_UNTIL", candidate_sql)
                self.assertIn("JOB_STATUS =", candidate_sql)
                for lock_sql in (first_lock_sql, second_lock_sql):
                    self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
                    self.assertNotIn("FETCH FIRST", lock_sql)
                    self.assertIn(
                        '"KBOT_KC_INGESTION_JOB".INGESTION_JOB_ID',
                        lock_sql,
                    )

    async def test_non_positive_limit_does_not_query_database(self):
        session = _Session([])

        claimed = await IngestionJobRepository(
            session
        ).claim_candidates(
            now=datetime.now(timezone.utc),
            limit=0,
        )

        self.assertEqual([], claimed)
        self.assertEqual([], session.statements)

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
