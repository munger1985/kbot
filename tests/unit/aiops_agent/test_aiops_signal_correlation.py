"""SignalEvent 与跨来源 Situation 确定性关联测试。"""

import unittest
from datetime import UTC, datetime, timedelta
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
from aiops_agent.repositories.messaging import OutboxRepository
from platform_core.identity import uuid7
from sqlalchemy.dialects import oracle


class SituationCorrelationTest(unittest.TestCase):
    def test_auto_run_idempotency_is_scoped_to_situation_and_agent(self) -> None:
        first = _auto_run_idempotency_key(
            situation_id="situation-1",
            agent_id="agent-1",
            signal_event_id="signal-1",
        )
        second = _auto_run_idempotency_key(
            situation_id="situation-2",
            agent_id="agent-1",
            signal_event_id="signal-1",
        )
        next_generation = _auto_run_idempotency_key(
            situation_id="situation-1",
            agent_id="agent-1",
            signal_event_id="signal-2",
        )

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, next_generation)
        self.assertEqual(
            first,
            _auto_run_idempotency_key(
                situation_id="situation-1",
                agent_id="agent-1",
                signal_event_id="signal-1",
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
            side_effect=([evidence_source_id], [target_id], [])
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
    async def test_warning_event_uses_agent_configured_warning_threshold(
        self,
    ) -> None:
        service = object.__new__(SignalEventIntakeService)
        target = SimpleNamespace(domain_id=7, target_id=uuid7())
        source_id = uuid7()
        binding = SimpleNamespace(
            agent_id=uuid7(),
            diagnostic_source_ids=(source_id,),
        )
        policy = SimpleNamespace(
            rules_json={
                "auto_observe_min_severity": "WARNING",
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

        result = await service._resolve_auto_agent(
            uow=uow,
            target=target,
            source_id=source_id,
            situation_id=uuid7(),
            severity="WARNING",
            fingerprint="f" * 64,
            now=datetime(2026, 9, 1, tzinfo=UTC),
        )

        self.assertEqual(binding, result[0])
        self.assertEqual(900, result[1])

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

        self.assertEqual(binding, result[0])
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

    async def test_recent_diagnosis_does_not_cool_down_new_situation(
        self,
    ) -> None:
        service = object.__new__(SignalEventIntakeService)
        target = SimpleNamespace(domain_id=7, target_id=uuid7())
        source_id = uuid7()
        current_situation_id = uuid7()
        previous_situation_id = uuid7()
        binding = SimpleNamespace(
            agent_id=uuid7(),
            diagnostic_source_ids=(source_id,),
        )
        policy = SimpleNamespace(
            rules_json={
                "auto_observe_min_severity": "CRITICAL",
                "alert_cooldown_seconds": 900,
            }
        )
        now = datetime(2026, 9, 1, tzinfo=UTC)
        uow = SimpleNamespace(
            agents=SimpleNamespace(
                resolve_auto_alert=AsyncMock(
                    return_value=(binding, policy)
                )
            ),
            runs=SimpleNamespace(
                get_latest_by_situation_correlation=AsyncMock(
                    return_value=SimpleNamespace(
                        situation_id=previous_situation_id,
                        created_at=now,
                    )
                )
            ),
        )

        result = await service._resolve_auto_agent(
            uow=uow,
            target=target,
            source_id=source_id,
            situation_id=current_situation_id,
            severity="CRITICAL",
            fingerprint="f" * 64,
            now=now,
        )

        self.assertEqual(binding, result[0])

    async def test_recent_diagnosis_cools_down_same_situation(self) -> None:
        service = object.__new__(SignalEventIntakeService)
        target = SimpleNamespace(domain_id=7, target_id=uuid7())
        source_id = uuid7()
        situation_id = uuid7()
        binding = SimpleNamespace(
            agent_id=uuid7(),
            diagnostic_source_ids=(source_id,),
        )
        policy = SimpleNamespace(
            rules_json={
                "auto_observe_min_severity": "CRITICAL",
                "alert_cooldown_seconds": 900,
            }
        )
        now = datetime(2026, 9, 1, tzinfo=UTC)
        uow = SimpleNamespace(
            agents=SimpleNamespace(
                resolve_auto_alert=AsyncMock(
                    return_value=(binding, policy)
                )
            ),
            runs=SimpleNamespace(
                get_latest_by_situation_correlation=AsyncMock(
                    return_value=SimpleNamespace(
                        situation_id=situation_id,
                        created_at=now,
                    )
                )
            ),
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
                now=now,
            )

        self.assertIsNone(result)
        self.assertEqual("COOLDOWN_ACTIVE", log.bind.call_args.kwargs["reason"])

    async def test_auto_run_deduplicates_in_progress_outbox(self) -> None:
        for status in ("PENDING", "PUBLISHING", "RETRY_WAIT"):
            with self.subTest(status=status):
                service, uow, context = self._auto_run_enqueue_context(
                    latest=SimpleNamespace(
                        outbox_id=uuid7(),
                        status=status,
                    )
                )
                with patch(
                    "aiops_agent.application.diagnostic_sources."
                    "webhook_intake.logger"
                ) as log:
                    await service._enqueue_auto_run(
                        uow=uow,
                        cooldown_seconds=900,
                        **context,
                    )

                uow.outbox.add.assert_not_awaited()
                self.assertEqual(
                    "OUTBOX_IN_PROGRESS",
                    log.bind.call_args.kwargs["reason"],
                )
                self.assertEqual(
                    status,
                    log.bind.call_args.kwargs["outbox_status"],
                )

    async def test_auto_run_deduplicates_published_outbox_during_cooldown(
        self,
    ) -> None:
        now = datetime(2026, 9, 2, tzinfo=UTC)
        service, uow, context = self._auto_run_enqueue_context(
            now=now,
            latest=SimpleNamespace(
                outbox_id=uuid7(),
                status="PUBLISHED",
                published_at=now - timedelta(seconds=899),
                updated_at=now,
                created_at=now,
            ),
        )

        with patch(
            "aiops_agent.application.diagnostic_sources.webhook_intake.logger"
        ) as log:
            await service._enqueue_auto_run(
                uow=uow,
                cooldown_seconds=900,
                **context,
            )

        uow.outbox.add.assert_not_awaited()
        self.assertEqual(
            "OUTBOX_COOLDOWN_ACTIVE",
            log.bind.call_args.kwargs["reason"],
        )

    async def test_auto_run_requeues_published_outbox_after_cooldown(
        self,
    ) -> None:
        now = datetime(2026, 9, 2, tzinfo=UTC)
        service, uow, context = self._auto_run_enqueue_context(
            now=now,
            latest=SimpleNamespace(
                outbox_id=uuid7(),
                status="PUBLISHED",
                published_at=now - timedelta(seconds=900),
                updated_at=now,
                created_at=now,
            ),
        )

        await service._enqueue_auto_run(
            uow=uow,
            cooldown_seconds=900,
            **context,
        )

        created = uow.outbox.add.await_args.args[0]
        self.assertIn(
            f":signal:{context['event_entity'].signal_event_id}",
            created.idempotency_key,
        )

    async def test_auto_run_requeues_after_failed_outbox(self) -> None:
        service, uow, context = self._auto_run_enqueue_context(
            latest=SimpleNamespace(
                outbox_id=uuid7(),
                status="FAILED",
            )
        )

        with patch(
            "aiops_agent.application.diagnostic_sources.webhook_intake.logger"
        ) as log:
            await service._enqueue_auto_run(
                uow=uow,
                cooldown_seconds=900,
                **context,
            )

        uow.outbox.add.assert_awaited_once()
        self.assertEqual(
            "PREVIOUS_OUTBOX_FAILED",
            log.bind.call_args.kwargs["reason"],
        )

    @staticmethod
    def _auto_run_enqueue_context(*, latest=None, now=None):
        service = object.__new__(SignalEventIntakeService)
        current_time = now or datetime(2026, 9, 2, tzinfo=UTC)
        target = SimpleNamespace(
            domain_id=7,
            target_id=uuid7(),
        )
        situation = SimpleNamespace(
            situation_id=uuid7(),
            severity="CRITICAL",
        )
        event_entity = SimpleNamespace(
            signal_event_id=uuid7(),
            diagnostic_source_id=uuid7(),
            occurred_at=current_time,
        )
        outbox = SimpleNamespace(
            get_latest_by_idempotency_prefix=AsyncMock(
                return_value=latest
            ),
            get_by_idempotency=AsyncMock(return_value=None),
            add=AsyncMock(),
        )
        return service, SimpleNamespace(outbox=outbox), {
            "target": target,
            "situation": situation,
            "event_entity": event_entity,
            "agent_id": uuid7(),
            "trace_id": "trace-auto-alert",
            "now": current_time,
        }

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
    async def test_situation_source_summary_uses_bounded_analytic_projection(
        self,
    ) -> None:
        result = Mock()
        result.mappings.return_value.all.return_value = []
        session = Mock()
        session.execute = AsyncMock(return_value=result)
        repository = SituationRepository(session)

        rows = await repository.summarize_sources_for_situation(
            situation_id=uuid7()
        )

        self.assertEqual([], rows)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=oracle.dialect())).upper()
        self.assertIn("ROW_NUMBER() OVER", sql)
        self.assertIn("COUNT(", sql)
        self.assertIn("PARTITION BY", sql)
        self.assertIn("KBOT_OPS_DIAGNOSTIC_SOURCE", sql)
        self.assertIn("SOURCE_RANK =", sql)

    async def test_latest_auto_run_outbox_query_accepts_legacy_and_generation_keys(
        self,
    ) -> None:
        result = Mock()
        result.scalar_one_or_none.return_value = None
        session = Mock()
        session.execute = AsyncMock(return_value=result)
        repository = OutboxRepository(session)

        await repository.get_latest_by_idempotency_prefix(
            aggregate_type="SITUATION",
            aggregate_id=uuid7(),
            idempotency_prefix="situation:s1:agent:a1:observe-run",
            event_type="OPS_SITUATION_AUTO_RUN_REQUESTED",
        )

        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=oracle.dialect())).upper()
        self.assertIn("IDEMPOTENCY_KEY =", sql)
        self.assertIn("IDEMPOTENCY_KEY LIKE", sql)
        self.assertIn("AGGREGATE_TYPE =", sql)
        self.assertIn("AGGREGATE_ID =", sql)
        self.assertIn("EVENT_TYPE =", sql)
        self.assertIn("ORDER BY", sql)

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
