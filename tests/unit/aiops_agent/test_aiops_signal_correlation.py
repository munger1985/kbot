"""SignalEvent 与跨来源 Situation 确定性关联测试。"""

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from aiops_agent.domain.evidence import (
    correlate_signal_event,
    validate_event_class_map,
)
from aiops_agent.repositories.monitoring import (
    DiagnosticSourceRepository,
    SituationRepository,
)
from platform_core.identity import uuid7
from sqlalchemy.dialects import oracle


class SituationCorrelationTest(unittest.TestCase):
    def test_explicit_mapping_correlates_different_source_classes(self) -> None:
        prometheus = correlate_signal_event(
            target_id="target-1",
            source_event_class="DatabaseDown",
            mapping_overrides={
                "event_class_map": {
                    "DatabaseDown": "database.unavailable"
                }
            },
        )
        zabbix = correlate_signal_event(
            target_id="target-1",
            source_event_class="zabbix.problem",
            mapping_overrides={
                "event_class_map": {
                    "zabbix.problem": "database.unavailable"
                }
            },
        )

        self.assertEqual(prometheus.correlation_hash, zabbix.correlation_hash)
        self.assertEqual("RULE", prometheus.method)
        self.assertEqual("database.unavailable", prometheus.canonical_event_class)

    def test_same_class_on_different_targets_does_not_correlate(self) -> None:
        first = correlate_signal_event(
            target_id="target-1",
            source_event_class="database.unavailable",
            mapping_overrides=None,
        )
        second = correlate_signal_event(
            target_id="target-2",
            source_event_class="database.unavailable",
            mapping_overrides=None,
        )

        self.assertNotEqual(first.correlation_hash, second.correlation_hash)
        self.assertEqual("EXACT", first.method)

    def test_invalid_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "规范的小写事件类别"):
            validate_event_class_map(
                {
                    "event_class_map": {
                        "DatabaseDown": "Database Unavailable"
                    }
                }
            )


class SituationStateRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_webhook_route_uses_enablement_not_connectivity_as_gate(
        self,
    ) -> None:
        result = Mock()
        result.scalar_one_or_none.return_value = None
        session = Mock()
        session.execute = AsyncMock(return_value=result)
        repository = DiagnosticSourceRepository(session)

        await repository.get_by_webhook_hash(
            webhook_key_hash="a" * 64,
            now=datetime(2026, 9, 1, tzinfo=UTC),
        )

        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=oracle.dialect())).upper()
        where_clause = sql.split("WHERE", 1)[1]
        self.assertIn("STATUS", where_clause)
        self.assertIn("WEBHOOK_KEY_HASH", where_clause)
        self.assertNotIn("CONNECTIVITY_STATUS", where_clause)

    async def test_latest_state_per_source_incident_controls_resolution(
        self,
    ) -> None:
        result = Mock()
        result.scalar_one_or_none.return_value = "OPEN"
        session = Mock()
        session.execute = AsyncMock(return_value=result)
        repository = SituationRepository(session)

        has_open = await repository.has_open_signal_state(
            situation_id=uuid7()
        )

        self.assertTrue(has_open)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=oracle.dialect())).upper()
        self.assertIn("ROW_NUMBER() OVER", sql)
        self.assertIn("PARTITION BY", sql)
        self.assertIn("DEDUP_HASH", sql)


if __name__ == "__main__":
    unittest.main()
