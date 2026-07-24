"""Agent Task 的 Oracle SKIP LOCKED 查询结构回归测试。"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from sqlalchemy.dialects import oracle

from agent_runtime.repositories.runtime import (
    AgentDelegationRepository,
    AgentTaskRepository,
)
from platform_core.identity import uuid7


class _Result:
    def __init__(self, value):
        self.value = value

    def scalars(self):
        return self.value

    def scalar_one_or_none(self):
        return self.value

    def one_or_none(self):
        return self.value


class _Session:
    def __init__(self, results):
        self.results = iter(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result(next(self.results))


class AgentTaskRepositoryOracleTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_expired_claim_locks_single_table_by_primary_key(self):
        now = datetime.now(timezone.utc)
        task_id = uuid7()
        run_id = uuid7()
        task = SimpleNamespace(
            task_id=task_id,
            run_id=run_id,
            status="RUNNING",
            lease_until=now - timedelta(seconds=1),
        )
        session = _Session(
            [
                [task_id],
                task,
                SimpleNamespace(
                    status="RUNNING",
                    deadline_at=now + timedelta(minutes=5),
                ),
            ]
        )

        claimed = await AgentTaskRepository(
            session
        ).claim_expired_lease(now=now)

        self.assertIs(task, claimed)
        candidate_sql = self._oracle_sql(session.statements[0])
        lock_sql = self._oracle_sql(session.statements[1])
        self.assertIn("JOIN", candidate_sql)
        self.assertIn("FETCH FIRST", candidate_sql)
        self.assertNotIn("FOR UPDATE", candidate_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
        self.assertNotIn("JOIN", lock_sql)
        self.assertNotIn("FETCH FIRST", lock_sql)

    async def test_delegation_poll_locks_by_primary_key(self):
        now = datetime.now(timezone.utc)
        first_id = uuid7()
        second_id = uuid7()
        delegation = SimpleNamespace(delegation_id=second_id)
        session = _Session([[first_id, second_id], None, delegation])

        claimed = await AgentDelegationRepository(
            session
        ).claim_poll_candidate(now=now)

        self.assertIs(delegation, claimed)
        candidate_sql = self._oracle_sql(session.statements[0])
        first_lock_sql = self._oracle_sql(session.statements[1])
        second_lock_sql = self._oracle_sql(session.statements[2])
        self.assertIn("FETCH FIRST", candidate_sql)
        self.assertNotIn("FOR UPDATE", candidate_sql)
        for lock_sql in (first_lock_sql, second_lock_sql):
            self.assertIn("FOR UPDATE SKIP LOCKED", lock_sql)
            self.assertNotIn("FETCH FIRST", lock_sql)
            self.assertIn(
                '"KBOT_AGENT_DELEGATION".DELEGATION_ID',
                lock_sql,
            )

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
