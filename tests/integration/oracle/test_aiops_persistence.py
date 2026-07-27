"""AIOps Persistence、Entity 和 UoW 的离线契约测试。"""

from __future__ import annotations

import inspect
from pathlib import Path
import re
import unittest

from aiops_agent.application.errors import UnitOfWorkStateError
from aiops_agent.entities import TargetEntity
from aiops_agent.persistence import AIOpsUnitOfWork, UnitOfWorkState
from aiops_agent.repositories import (
    AlertRepository,
    ChangeRepository,
    InboxRepository,
    InspectionRepository,
    MonitorSourceRepository,
    OpsRunRepository,
    OutboxRepository,
    PolicyRepository,
    TargetRepository,
)
from tests.acceptance.check_aiops_entity_schema import AIOPS_ENTITY_CLASSES
from tests.acceptance.check_oracle_schema import SERVICE_TABLES


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = ROOT / "database" / "oracle" / "aiops_agent"
REPOSITORIES = (
    TargetRepository,
    MonitorSourceRepository,
    PolicyRepository,
    AlertRepository,
    OpsRunRepository,
    ChangeRepository,
    InspectionRepository,
    InboxRepository,
    OutboxRepository,
)


def _ddl_columns() -> dict[str, set[str]]:
    sql = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SCHEMA_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    )
    result: dict[str, set[str]] = {}
    for table_name, body in re.findall(
        r"CREATE TABLE\s+([A-Z][A-Z0-9_]*)\s*\((.*?)\n\);",
        sql,
        flags=re.DOTALL,
    ):
        columns = {
            match.group(1)
            for line in body.splitlines()
            if (
                match := re.match(
                    r"\s{4}([A-Z][A-Z0-9_]*)\s+"
                    r"(?:RAW|NUMBER|VARCHAR2|CLOB|JSON|TIMESTAMP)\b",
                    line,
                )
            )
        }
        result[table_name] = columns
    return result


class _FakeTransaction:
    def __init__(self):
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


class _FakeSession:
    def __init__(self):
        self.transaction = _FakeTransaction()
        self.flushes = 0
        self.rollbacks = 0
        self.closes = 0

    async def begin(self) -> _FakeTransaction:
        return self.transaction

    async def flush(self) -> None:
        self.flushes += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def close(self) -> None:
        self.closes += 1


class AIOpsEntityContractTest(unittest.TestCase):
    def test_entities_own_exact_manifest_tables_and_columns(self) -> None:
        entities = {
            entity.__tablename__: {
                column.name.upper() for column in entity.__table__.columns
            }
            for entity in AIOPS_ENTITY_CLASSES
        }
        self.assertEqual(SERVICE_TABLES["aiops_agent"], set(entities))
        self.assertEqual(_ddl_columns(), entities)

    def test_entities_do_not_define_lazy_relationships(self) -> None:
        for entity in AIOPS_ENTITY_CLASSES:
            self.assertEqual([], list(entity.__mapper__.relationships))

    def test_repository_has_no_transaction_or_external_io_ownership(
        self,
    ) -> None:
        forbidden = (
            ".commit(",
            ".rollback(",
            "httpx",
            "requests",
            "model_serving",
            "knowledge_core",
        )
        for repository in REPOSITORIES:
            source = inspect.getsource(repository)
            for token in forbidden:
                self.assertNotIn(token, source, repository.__name__)

    def test_oracle_claims_fetch_one_row_on_the_server(self) -> None:
        claim_repositories = (
            OpsRunRepository,
            InspectionRepository,
            OutboxRepository,
        )
        source = "\n".join(
            inspect.getsource(repository)
            for repository in claim_repositories
        )
        self.assertGreaterEqual(source.count("FOR UPDATE OF"), 2)
        self.assertGreaterEqual(
            source.count("FETCH c_claim INTO :claimed_id"),
            2,
        )
        runtime_source = inspect.getsource(OpsRunRepository)
        self.assertIn("with_for_update(skip_locked=True)", runtime_source)
        self.assertIn("候选查询不持锁", runtime_source)


class AIOpsUnitOfWorkTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_commit_is_single_use_and_closes_repositories(
        self,
    ) -> None:
        session = _FakeSession()
        uow = AIOpsUnitOfWork(lambda: session)
        async with uow:
            repository = uow.targets
            self.assertIsNotNone(repository)
            await uow.commit()
            self.assertEqual(UnitOfWorkState.COMMITTED, uow.state)
            with self.assertRaises(UnitOfWorkStateError):
                await uow.commit()
            with self.assertRaises(UnitOfWorkStateError):
                await repository.add_target(TargetEntity())

        self.assertEqual(1, session.transaction.commits)
        self.assertEqual(0, session.rollbacks)
        self.assertEqual(1, session.closes)
        self.assertEqual(UnitOfWorkState.CLOSED, uow.state)

    async def test_normal_exit_without_commit_rolls_back(self) -> None:
        session = _FakeSession()
        uow = AIOpsUnitOfWork(lambda: session)
        async with uow:
            self.assertEqual(UnitOfWorkState.ACTIVE, uow.state)

        self.assertEqual(0, session.transaction.commits)
        self.assertEqual(1, session.rollbacks)
        self.assertEqual(1, session.closes)

    async def test_explicit_rollback_is_not_repeated_on_exit(self) -> None:
        session = _FakeSession()
        uow = AIOpsUnitOfWork(lambda: session)
        async with uow:
            await uow.rollback()
            self.assertEqual(UnitOfWorkState.ROLLED_BACK, uow.state)

        self.assertEqual(1, session.rollbacks)
        self.assertEqual(1, session.closes)


if __name__ == "__main__":
    unittest.main()
