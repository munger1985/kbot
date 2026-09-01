"""SignalEvent 与跨来源 Situation 确定性关联测试。"""

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aiops_agent.application.diagnostic_sources.webhook_intake import (
    SignalEventIntakeService,
    _auto_run_idempotency_key,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.domain.evidence import (
    correlate_signal_event,
    validate_event_class_map,
)
from aiops_agent.repositories.agent import AIOpsAgentRepository
from aiops_agent.repositories.monitoring import (
    DiagnosticSourceRepository,
    SituationRepository,
)
from platform_core.identity import uuid7
from sqlalchemy.dialects import oracle


class SituationCorrelationTest(unittest.TestCase):
    def test_auto_run_idempotency_is_scoped_to_signal(self) -> None:
        first = _auto_run_idempotency_key(
            situation_id="situation-1",
            signal_event_id="signal-1",
            agent_id="agent-1",
        )
        second = _auto_run_idempotency_key(
            situation_id="situation-1",
            signal_event_id="signal-2",
            agent_id="agent-1",
        )

        self.assertNotEqual(first, second)
        self.assertEqual(
            first,
            _auto_run_idempotency_key(
                situation_id="situation-1",
                signal_event_id="signal-1",
                agent_id="agent-1",
            ),
        )

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


class AutoAlertAgentRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_target_agent_is_fallback_when_transport_source_is_not_selected(
        self,
    ) -> None:
        alertmanager_source_id = uuid7()
        evidence_source_id = uuid7()
        target_id = uuid7()
        agent = SimpleNamespace(
            agent_id=uuid7(),
            status="ACTIVE",
            row_version=1,
        )
        version = SimpleNamespace(
            agent_version_id=uuid7(),
            policy_id=uuid7(),
            instruction=None,
        )
        policy = SimpleNamespace(
            policy_id=version.policy_id,
            rules_json={"auto_alert_enabled": True},
        )
        exact = MagicMock()
        exact.__iter__.return_value = iter(())
        fallback = MagicMock()
        fallback.__iter__.return_value = iter(((agent, version, policy),))
        session = MagicMock()
        session.execute = AsyncMock(side_effect=(exact, fallback))
        session.scalars = AsyncMock(
            side_effect=([evidence_source_id], [target_id])
        )
        repository = AIOpsAgentRepository(session, lambda: None)

        resolved = await repository.resolve_auto_alert(
            domain_id=7,
            source_id=alertmanager_source_id,
            target_id=target_id,
        )

        self.assertIsNotNone(resolved)
        binding, resolved_policy = resolved
        self.assertEqual(agent.agent_id, binding.agent_id)
        self.assertEqual((evidence_source_id,), binding.diagnostic_source_ids)
        self.assertEqual(policy, resolved_policy)
        self.assertEqual(2, session.execute.await_count)
        exact_sql = str(
            session.execute.await_args_list[0].args[0].compile(
                dialect=oracle.dialect()
            )
        ).upper()
        fallback_sql = str(
            session.execute.await_args_list[1].args[0].compile(
                dialect=oracle.dialect()
            )
        ).upper()
        self.assertIn("KBOT_OPS_AGENT_VERSION_SOURCE", exact_sql)
        self.assertNotIn("KBOT_OPS_AGENT_VERSION_SOURCE", fallback_sql)
        self.assertIn("KBOT_OPS_AGENT_VERSION_TARGET", fallback_sql)


class SignalIntakeReceiptTest(unittest.IsolatedAsyncioTestCase):
    async def test_auto_agent_logs_target_fallback_selection(self) -> None:
        service = object.__new__(SignalEventIntakeService)
        target = SimpleNamespace(domain_id=7, target_id=uuid7())
        alertmanager_source_id = uuid7()
        binding = SimpleNamespace(
            agent_id=uuid7(),
            diagnostic_source_ids=(uuid7(),),
        )
        policy = SimpleNamespace(
            rules_json={
                "auto_observe_min_severity": "CRITICAL",
                "alert_cooldown_seconds": 900,
            }
        )
        uow = SimpleNamespace(
            agents=SimpleNamespace(
                resolve_auto_alert=AsyncMock(
                    return_value=(binding, policy)
                )
            ),
            runs=SimpleNamespace(
                get_latest_by_situation_correlation=AsyncMock(
                    return_value=None
                )
            ),
        )

        with patch(
            "aiops_agent.application.diagnostic_sources.webhook_intake.logger"
        ) as log:
            result = await service._resolve_auto_agent(
                uow=uow,
                target=target,
                source_id=alertmanager_source_id,
                situation_id=uuid7(),
                severity="CRITICAL",
                fingerprint="f" * 64,
                now=datetime(2026, 9, 1, tzinfo=UTC),
            )

        self.assertEqual(binding, result)
        self.assertEqual(
            "TARGET_AGENT_FALLBACK",
            log.bind.call_args.kwargs["reason"],
        )

    async def test_auto_agent_rejection_records_structured_reason(self) -> None:
        service = object.__new__(SignalEventIntakeService)
        target = SimpleNamespace(domain_id=7, target_id=uuid7())
        source_id = uuid7()
        situation_id = uuid7()
        uow = SimpleNamespace(
            agents=SimpleNamespace(
                resolve_auto_alert=AsyncMock(return_value=None)
            )
        )

        with patch(
            "aiops_agent.application.diagnostic_sources.webhook_intake.logger"
        ) as log:
            result = await service._resolve_auto_agent(
                uow=uow,
                target=target,
                source_id=source_id,
                situation_id=situation_id,
                severity="CRITICAL",
                fingerprint="f" * 64,
                now=datetime(2026, 9, 1, tzinfo=UTC),
            )

        self.assertIsNone(result)
        self.assertEqual(
            "NO_ELIGIBLE_AGENT",
            log.bind.call_args.kwargs["reason"],
        )
        self.assertEqual(
            str(situation_id),
            log.bind.call_args.kwargs["situation_id"],
        )
        log.bind.return_value.info.assert_called_once()

    async def test_duplicate_unmatched_target_remains_rejected(self) -> None:
        uow = SimpleNamespace(
            situations=SimpleNamespace(
                list_signal_event_ids_by_inbox=AsyncMock()
            )
        )
        inbox = SimpleNamespace(
            inbox_id=uuid7(),
            status="IGNORED",
            error_code="SOURCE_TARGET_NOT_FOUND",
        )

        with self.assertRaises(AIOpsApplicationError) as raised:
            await SignalEventIntakeService._duplicate_receipt(uow, inbox)

        self.assertEqual("SOURCE_TARGET_NOT_FOUND", raised.exception.code)
        self.assertEqual(422, raised.exception.status_code)
        uow.situations.list_signal_event_ids_by_inbox.assert_not_awaited()

    async def test_duplicate_processed_event_is_accepted(self) -> None:
        signal_event_id = uuid7()
        uow = SimpleNamespace(
            situations=SimpleNamespace(
                list_signal_event_ids_by_inbox=AsyncMock(
                    return_value=[signal_event_id]
                )
            )
        )
        inbox = SimpleNamespace(
            inbox_id=uuid7(),
            status="PROCESSED",
            error_code=None,
        )

        receipt = await SignalEventIntakeService._duplicate_receipt(
            uow, inbox
        )

        self.assertTrue(receipt.accepted)
        self.assertTrue(receipt.duplicate)
        self.assertEqual((signal_event_id,), receipt.signal_event_ids)


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
