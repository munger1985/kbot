"""步骤 10A 巡检调度、Fire 与收敛测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.scheduling import (
    AIOpsInspectionScheduler,
    resolve_due_schedule,
)
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.identity import uuid7


class ScheduleResolverTest(unittest.TestCase):
    def test_daily_window_is_half_open_previous_business_day(
        self,
    ) -> None:
        due = datetime(2026, 7, 24, 1, 0, tzinfo=UTC)
        result = resolve_due_schedule(
            cron_expression="0 9 * * *",
            timezone_name="Asia/Shanghai",
            schedule_type="DAILY",
            due_at=due,
            now=due + timedelta(seconds=10),
            misfire_policy="LATEST_ONLY",
            misfire_grace_seconds=60,
            resolver_version="1.0.0",
        )
        self.assertFalse(result.skipped)
        self.assertEqual(
            result.period_end,
            datetime(2026, 7, 23, 16, 0, tzinfo=UTC),
        )
        self.assertEqual(
            result.period_end - result.period_start,
            timedelta(days=1),
        )

    def test_skip_misfire_advances_without_replay(self) -> None:
        due = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
        now = datetime(2026, 7, 24, 2, 0, tzinfo=UTC)
        result = resolve_due_schedule(
            cron_expression="0 9 * * *",
            timezone_name="Asia/Shanghai",
            schedule_type="DAILY",
            due_at=due,
            now=now,
            misfire_policy="SKIP",
            misfire_grace_seconds=60,
            resolver_version="1.0.0",
        )
        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "MISFIRE_SKIPPED")
        self.assertGreater(result.next_run_at, now)

    def test_latest_only_collapses_missed_occurrences(self) -> None:
        due = datetime(2026, 7, 20, 1, 0, tzinfo=UTC)
        now = datetime(2026, 7, 24, 2, 0, tzinfo=UTC)
        result = resolve_due_schedule(
            cron_expression="0 9 * * *",
            timezone_name="Asia/Shanghai",
            schedule_type="DAILY",
            due_at=due,
            now=now,
            misfire_policy="LATEST_ONLY",
            misfire_grace_seconds=60,
            resolver_version="1.0.0",
        )
        self.assertEqual(
            result.scheduled_for,
            datetime(2026, 7, 24, 1, 0, tzinfo=UTC),
        )
        self.assertEqual(
            result.resolution["occurrences_collapsed"], 5
        )


class InspectionSchedulerTest(unittest.TestCase):
    def test_due_plan_atomically_creates_fire_and_target_requests(
        self,
    ) -> None:
        now = datetime(2026, 7, 24, 1, 0, 10, tzinfo=UTC)
        plan = SimpleNamespace(
            inspection_plan_id=uuid7(),
            domain_id=200,
            display_name="数据库日报",
            schedule_type="DAILY",
            cron_expression="0 9 * * *",
            timezone="Asia/Shanghai",
            template_id="database_daily",
            template_version="1.0.0",
            timeout_seconds=3600,
            overlap_policy="SKIP",
            misfire_policy="LATEST_ONLY",
            schedule_resolver_version="1.0.0",
            next_run_at=datetime(2026, 7, 24, 1, 0, tzinfo=UTC),
            row_version=3,
        )
        targets = [
            SimpleNamespace(
                target_id=uuid7(),
                template_overrides_json=None,
            ),
            SimpleNamespace(
                target_id=uuid7(),
                template_overrides_json={"thresholds": {"cpu": 90}},
            ),
        ]
        fires = []
        messages = []

        async def add_fire(entity):
            fires.append(entity)
            return entity

        async def add_outbox(entity):
            messages.append(entity)
            return entity

        inspections = SimpleNamespace(
            claim_due_plan=AsyncMock(return_value=plan),
            list_active_targets=AsyncMock(return_value=targets),
            list_open_fires=AsyncMock(return_value=[]),
            add_fire=AsyncMock(side_effect=add_fire),
            advance_claimed_plan=AsyncMock(return_value=True),
            find_reconcilable_fire=AsyncMock(return_value=None),
        )
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now)
            ),
            inspections=inspections,
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=add_outbox)
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        scheduler = AIOpsInspectionScheduler(
            uow_factory=lambda: context,
            scheduler_id="scheduler-test",
            system_agent_id=uuid7(),
            lease_seconds=120,
            interval_seconds=30,
            misfire_grace_seconds=60,
        )
        worked = asyncio.run(scheduler.run_once())
        self.assertTrue(worked)
        self.assertEqual(len(fires), 1)
        self.assertEqual(fires[0].status, "RUNNING")
        self.assertEqual(fires[0].target_count, 2)
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            {item.event_type for item in messages},
            {"OPS_INSPECTION_RUN_REQUESTED"},
        )
        inspections.advance_claimed_plan.assert_awaited_once()
        uow.commit.assert_awaited_once()

    def test_running_fire_aggregates_mixed_terminal_runs(self) -> None:
        now = datetime(2026, 7, 24, 2, 0, tzinfo=UTC)
        fire = SimpleNamespace(
            inspection_fire_id=uuid7(),
            inspection_plan_id=uuid7(),
            status="RUNNING",
            target_count=2,
            run_count=0,
            completed_count=0,
            failed_count=0,
            completed_at=None,
            updated_at=now,
        )
        runs = [
            SimpleNamespace(status="COMPLETED"),
            SimpleNamespace(status="FAILED"),
        ]
        inspections = SimpleNamespace(
            claim_due_plan=AsyncMock(return_value=None),
            find_reconcilable_fire=AsyncMock(return_value=fire),
            get_fire=AsyncMock(return_value=fire),
            list_runs_for_fire=AsyncMock(return_value=runs),
            list_run_request_events_for_fire=AsyncMock(
                return_value=[
                    SimpleNamespace(status="PUBLISHED"),
                    SimpleNamespace(status="PUBLISHED"),
                ]
            ),
        )
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now)
            ),
            inspections=inspections,
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        scheduler = AIOpsInspectionScheduler(
            uow_factory=lambda: context,
            scheduler_id="scheduler-test",
            system_agent_id=uuid7(),
            lease_seconds=120,
            interval_seconds=30,
            misfire_grace_seconds=60,
        )
        worked = asyncio.run(scheduler.run_once())
        self.assertTrue(worked)
        self.assertEqual(fire.status, "PARTIAL")
        self.assertEqual(fire.completed_count, 1)
        self.assertEqual(fire.failed_count, 1)

    def test_run_request_preserves_fire_and_report_window(self) -> None:
        runtime = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=AsyncMock(),
        )
        fire_id = uuid7()
        target_id = uuid7()
        payload = {
            "inspection_fire_id": str(fire_id),
                        "domain_id": 200,
            "actor_id": "system:inspection-scheduler",
            "agent_id": str(uuid7()),
            "target_id": str(target_id),
            "template_id": "database_daily",
            "template_version": "1.0.0",
            "schedule_type": "DAILY",
            "timezone": "Asia/Shanghai",
            "period_start": "2026-07-22T16:00:00+00:00",
            "period_end": "2026-07-23T16:00:00+00:00",
            "timeout_seconds": 3600,
            "template_overrides": {},
            "trace_id": "trace-inspection",
        }
        asyncio.run(
            sink.publish("OPS_INSPECTION_RUN_REQUESTED", payload)
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual(command.inspection_fire_id, fire_id)
        self.assertEqual(command.trigger_type, "SCHEDULE")
        self.assertEqual(
            command.blueprint_id, "database.diagnostic-baseline"
        )
        self.assertEqual(
            command.client_metadata["inspection"]["timezone"],
            "Asia/Shanghai",
        )


if __name__ == "__main__":
    unittest.main()
