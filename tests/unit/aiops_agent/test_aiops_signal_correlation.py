"""SignalEvent 与跨来源 Situation 确定性关联测试。"""

import unittest
from unittest.mock import AsyncMock, Mock

from sqlalchemy.dialects import oracle

from aiops_agent.domain.evidence import (
    correlate_signal_event,
    validate_event_class_map,
)
from aiops_agent.repositories.monitoring import SituationRepository
from platform_core.identity import uuid7


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
