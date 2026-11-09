"""当前/历史告警的时间、状态、分页与信号恢复语义回归。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from aiops_agent.application.alert_query import AlertQueryParameters, signal_status_label
from aiops_agent.application.configuration.common import SignedCursorCodec
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.turn_capabilities import RuntimeQueryAdapter
from aiops_agent.repositories.monitoring import SituationRepository
from platform_core.contracts.aiops import AIOpsCapability, TimeRangeV1
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 9, 2, 2, 21, tzinfo=UTC)


def _row(title, status, observed_at):
    row = SimpleNamespace(situation_id=uuid7(), target_id=uuid7(), title=title,
        summary=title, status=status, severity="WARNING", last_observed_at=observed_at)
    row.model_dump = lambda **_: {"title": title, "status": status}
    return row


class AlertQueryParametersTest(unittest.TestCase):
    def test_current_defaults_to_recent_hour(self):
        query = AlertQueryParameters()
        self.assertEqual("CURRENT", query.mode)
        self.assertEqual((NOW - timedelta(hours=1), NOW), query.time_bounds(None, now=NOW))

    def test_history_defaults_to_recent_day(self):
        query = AlertQueryParameters(mode="HISTORY", status="RESOLVED")
        self.assertEqual((NOW - timedelta(days=1), NOW), query.time_bounds(None, now=NOW))

    def test_explicit_absolute_and_relative_ranges_are_used(self):
        query = AlertQueryParameters(mode="HISTORY")
        start, end = NOW - timedelta(days=20), NOW - timedelta(days=19)
        self.assertEqual((start, end), query.time_bounds(TimeRangeV1(start_at=start, end_at=end), now=NOW))
        self.assertEqual((NOW - timedelta(minutes=15), NOW),
            query.time_bounds(TimeRangeV1(relative_window_seconds=900), now=NOW))

    def test_invalid_or_contradictory_conditions_are_not_silently_ignored(self):
        for parameters in ({"status": "RESOLVED"}, {"status": "CLOSED"}, {"mode": "ALL"},
            {"limit": 0}, {"limit": 101}, {"severity": "guessed"}, {"unknown": True}):
            with self.subTest(parameters=parameters), self.assertRaises(ValidationError):
                AlertQueryParameters.model_validate(parameters)

    def test_signal_recovery_is_not_human_resolution(self):
        self.assertEqual("告警已恢复", signal_status_label("RESOLVED"))
        self.assertNotIn("已解决", signal_status_label("RESOLVED"))


class AlertQueryAdapterTest(unittest.IsolatedAsyncioTestCase):
    async def _execute(self, rows=(), *, parameters=None, time_range=None, has_more=False):
        calls = []

        async def list_situations(**kwargs):
            calls.append(kwargs)
            selected = tuple(row for row in rows
                if kwargs["observed_since"] <= row.last_observed_at < kwargs["observed_until"]
                and row.status not in kwargs["exclude_statuses"]
                and (kwargs["status"] is None or row.status == kwargs["status"])
                and (kwargs["severity"] is None or row.severity == kwargs["severity"]))
            return SimpleNamespace(items=selected[:kwargs["limit"]], has_more=has_more)

        runtime = SimpleNamespace(list_situations=list_situations)
        plan = SimpleNamespace(time_range=time_range, steps=(SimpleNamespace(
            capability=AIOpsCapability.ALERT_QUERY, parameters=parameters or {}),))
        target_id = uuid7()
        internal = {"execution_targets": [{"target_id": str(target_id), "readiness": "READY"}]}
        with patch("aiops_agent.application.turn_capabilities.datetime") as clock:
            clock.now.return_value = NOW
            internal["runtime_context"] = {"domain_id": 7, "actor_id": "test-dba",
                "caller_service": "aiops-worker", "request_id": "test-query", "trace_id": "test-query"}
            result = await RuntimeQueryAdapter(runtime_service=runtime).execute(
                capability=AIOpsCapability.ALERT_QUERY, plan=plan, scope=None, internal_scope=internal)
        return result, calls

    async def test_current_health_excludes_old_recovered_and_stale_open_records(self):
        rows = (_row("新活动告警", "OPEN", NOW - timedelta(minutes=2)),
            _row("旧抓取失败", "RESOLVED", datetime(2026, 10, 21, tzinfo=UTC)),
            _row("旧验收告警", "OPEN", datetime(2026, 10, 11, tzinfo=UTC)),
            _row("窗口内恢复", "RESOLVED", NOW - timedelta(minutes=1)),
            _row("窗口内关闭", "CLOSED", NOW - timedelta(minutes=1)))
        result, calls = await self._execute(rows)
        self.assertTrue(result.supported)
        self.assertEqual(1, len(result.evidence))
        self.assertIn("新活动告警", result.markdown)
        for title in ("旧抓取失败", "旧验收告警", "窗口内恢复", "窗口内关闭"):
            self.assertNotIn(title, result.markdown)
        self.assertEqual(("RESOLVED", "CLOSED"), calls[0]["exclude_statuses"])
        self.assertEqual(NOW - timedelta(hours=1), calls[0]["observed_since"])
        self.assertEqual(NOW, calls[0]["observed_until"])

    async def test_history_can_query_old_recovered_alerts_explicitly(self):
        observed = datetime(2026, 10, 21, 11, 28, 45, tzinfo=UTC)
        result, calls = await self._execute((_row("旧抓取失败", "RESOLVED", observed),),
            parameters={"mode": "HISTORY", "status": "RESOLVED", "severity": "WARNING", "limit": 10},
            time_range=TimeRangeV1(start_at=observed - timedelta(hours=1), end_at=observed + timedelta(hours=1)))
        self.assertIn("历史告警", result.markdown)
        self.assertIn("旧抓取失败", result.markdown)
        self.assertIn("告警已恢复", result.markdown)
        self.assertNotIn("已解决", result.markdown)
        self.assertEqual((), calls[0]["exclude_statuses"])
        self.assertEqual("RESOLVED", calls[0]["status"])
        self.assertEqual("WARNING", calls[0]["severity"])

    async def test_empty_result_is_scoped_not_a_claim_of_current_health(self):
        result, _ = await self._execute()
        self.assertTrue(result.supported)
        self.assertIn("上述时间与状态条件", result.markdown)
        self.assertNotIn("没有告警记录", result.markdown)
        self.assertNotIn("健康正常", result.markdown)

    async def test_window_edges_are_half_open_and_status_severity_are_preserved(self):
        rows = (_row("起点活动", "OPEN", NOW - timedelta(hours=1)),
            _row("终点活动", "OPEN", NOW),
            _row("已确认活动", "ACKNOWLEDGED", NOW - timedelta(minutes=1)))
        result, _ = await self._execute(rows)
        self.assertIn("起点活动", result.markdown)
        self.assertNotIn("终点活动", result.markdown)
        self.assertIn("已确认活动", result.markdown)
        result, calls = await self._execute(rows, parameters={"status": "ACKNOWLEDGED", "severity": "WARNING"})
        self.assertEqual(1, len(result.evidence))
        self.assertNotIn("起点活动", result.markdown)
        self.assertEqual("ACKNOWLEDGED", calls[0]["status"])
        self.assertEqual("WARNING", calls[0]["severity"])

    async def test_invalid_filter_never_queries_all_records(self):
        result, calls = await self._execute(parameters={"status": "RESOLVED"})
        self.assertFalse(result.supported)
        self.assertEqual("ALERT_QUERY_INVALID_FILTER", result.gap_code)
        self.assertEqual([], calls)

    async def test_truncated_query_does_not_claim_complete_coverage(self):
        result, _ = await self._execute(has_more=True)
        self.assertEqual("ALERT_QUERY_TRUNCATED", result.gap_code)
        self.assertIn("未展示", result.markdown)


class AlertQueryPersistenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_repository_filters_observation_and_status_before_pagination(self):
        session = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: ())))
        await SituationRepository(session).page_situations(domain_id=7,
            target_id=uuid7(), exclude_statuses=("RESOLVED", "CLOSED"), severity="WARNING",
            observed_since=NOW - timedelta(hours=1), observed_until=NOW, limit=21)
        statement = session.execute.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        sql = str(compiled).replace('"', "").lower()
        self.assertIn("status not in ('resolved', 'closed')", sql)
        self.assertIn("last_observed_at >=", sql)
        self.assertIn("last_observed_at <", sql)
        self.assertLess(sql.index("last_observed_at >="), sql.index("limit"))
        self.assertIn("severity = 'warning'", sql)

    async def test_runtime_binds_time_and_status_filters_to_signed_cursor(self):
        repository = SimpleNamespace(page_situations=AsyncMock(return_value=[]))
        uow = AsyncMock()
        uow.__aenter__.return_value = SimpleNamespace(situations=repository)
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = lambda: uow
        runtime._cursor_codec = SignedCursorCodec(secret="alert-query-test-key-32-characters-long", ttl_seconds=3600)
        scope = SimpleNamespace(tenant_id=uuid7(), domain_id=7, principal_id=uuid7())
        target = uuid7()
        start = NOW - timedelta(hours=1)
        filters = {"target_id": str(target), "agent_id": None, "status": None, "severity": None,
            "exclude_statuses": ["CLOSED", "RESOLVED"],
            "observed_since": start.isoformat(), "observed_until": NOW.isoformat()}
        cursor = runtime._cursor_codec.encode(scope=scope, sort_at=NOW, resource_id=uuid7(), filters=filters)
        kwargs = dict(scope=scope, target_id=target, agent_id=None, status=None, severity=None,
            cursor=cursor, limit=20, exclude_statuses=("RESOLVED", "CLOSED"),
            observed_since=start, observed_until=NOW)
        result = await runtime.list_situations(**kwargs)
        self.assertEqual((), result.items)
        self.assertEqual(start, repository.page_situations.await_args.kwargs["observed_since"])
        self.assertEqual(("CLOSED", "RESOLVED"), repository.page_situations.await_args.kwargs["exclude_statuses"])
        with self.assertRaises(AIOpsApplicationError):
            await runtime.list_situations(**{**kwargs, "observed_since": start - timedelta(days=1)})
        repository.page_situations.assert_awaited_once()

    async def test_invalid_time_range_does_not_query_database(self):
        runtime = object.__new__(AIOpsRuntimeService)
        runtime._uow_factory = lambda: self.fail("无效时间范围不应访问数据库")
        with self.assertRaises(AIOpsApplicationError):
            await runtime.list_situations(scope=SimpleNamespace(), target_id=None,
                agent_id=None, status=None, severity=None, cursor=None, limit=20,
                observed_since=NOW, observed_until=NOW)


if __name__ == "__main__":
    unittest.main()
