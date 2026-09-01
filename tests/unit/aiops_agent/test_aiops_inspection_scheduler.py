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
from aiops_agent.application.turns import ConversationTurnService
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
    def test_agent_task_creates_standard_turns_for_current_targets(self) -> None:
        fire_id = uuid7()
        agent_id = uuid7()
        version_id = uuid7()
        target_ids = [uuid7(), uuid7()]
        fire = SimpleNamespace(
            plan_snapshot_json={"agent_version_id": None},
            target_count=0,
            updated_at=None,
        )
        conversations = []

        async def add_conversation(entity):
            conversations.append(entity)
            return entity

        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                get_fire=AsyncMock(return_value=fire),
            ),
            conversations=SimpleNamespace(
                list_for_inspection_fire=AsyncMock(return_value=[]),
                add_conversation=AsyncMock(side_effect=add_conversation),
            ),
            agents=SimpleNamespace(
                get=AsyncMock(
                    return_value=SimpleNamespace(
                        agent_id=agent_id,
                        status="ACTIVE",
                        current_version_id=version_id,
                    )
                ),
                version=AsyncMock(
                    return_value=SimpleNamespace(
                        agent_version_id=version_id
                    )
                ),
                active_version_target_ids=AsyncMock(
                    return_value=target_ids
                ),
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = ConversationTurnService(uow_factory=lambda: context)
        service._require_existing_target = AsyncMock()
        service._create_turn = AsyncMock(
            return_value={"status": "QUEUED"}
        )
        payload = {
            "inspection_fire_id": str(fire_id),
            "domain_id": 200,
            "actor_id": "system:inspection-scheduler",
            "agent_id": str(agent_id),
            "plan_display_name": "数据库日报",
            "template_id": "database_daily",
            "template_version": "1.0.0",
            "schedule_type": "DAILY",
            "timezone": "Asia/Shanghai",
            "period_start": "2026-07-22T16:00:00+00:00",
            "period_end": "2026-07-23T16:00:00+00:00",
            "timeout_seconds": 3600,
            "trace_id": "trace-inspection",
        }

        result = asyncio.run(service.start_scheduled_inspection(payload))

        self.assertEqual(result["conversation_count"], 2)
        self.assertEqual(len(conversations), 2)
        self.assertTrue(
            all(item.source_type == "INSPECTION" for item in conversations)
        )
        self.assertEqual(fire.target_count, 2)
        self.assertEqual(
            fire.plan_snapshot_json["agent_version_id"], str(version_id)
        )
        self.assertEqual(service._create_turn.await_count, 2)
        for call in service._create_turn.await_args_list:
            execution = call.kwargs["execution_context"]
            self.assertEqual(execution["trigger_type"], "SCHEDULE")
            self.assertEqual(execution["inspection_fire_id"], str(fire_id))
        uow.commit.assert_awaited_once()

    def test_due_plan_atomically_creates_fire_and_single_agent_request(
        self,
    ) -> None:
        now = datetime(2026, 7, 24, 1, 0, 10, tzinfo=UTC)
        agent_id = uuid7()
        plan = SimpleNamespace(
            inspection_plan_id=uuid7(),
            domain_id=200,
            agent_id=agent_id,
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
            lease_seconds=120,
            interval_seconds=30,
            misfire_grace_seconds=60,
        )
        worked = asyncio.run(scheduler.run_once())
        self.assertTrue(worked)
        self.assertEqual(len(fires), 1)
        self.assertEqual(fires[0].status, "RUNNING")
        self.assertEqual(fires[0].target_count, 0)
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            {item.event_type for item in messages},
            {"OPS_INSPECTION_AGENT_REQUESTED"},
        )
        self.assertEqual(
            {item.payload_json["agent_id"] for item in messages},
            {str(agent_id)},
        )
        self.assertNotIn("agent_version_id", messages[0].payload_json)
        self.assertNotIn("target_id", messages[0].payload_json)
        inspections.advance_claimed_plan.assert_awaited_once()
        uow.commit.assert_awaited_once()

    def test_running_fire_aggregates_agent_turn_terminals(self) -> None:
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
        runs = [SimpleNamespace(status="COMPLETED")]
        turns = [
            SimpleNamespace(status="COMPLETED"),
            SimpleNamespace(status="PARTIAL"),
        ]
        inspections = SimpleNamespace(
            claim_due_plan=AsyncMock(return_value=None),
            find_reconcilable_fire=AsyncMock(return_value=fire),
            get_fire=AsyncMock(return_value=fire),
            list_runs_for_fire=AsyncMock(return_value=runs),
            list_agent_request_events_for_fire=AsyncMock(
                return_value=[
                    SimpleNamespace(status="PUBLISHED"),
                ]
            ),
            list_turns_for_fire=AsyncMock(return_value=turns),
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
            lease_seconds=120,
            interval_seconds=30,
            misfire_grace_seconds=60,
        )
        worked = asyncio.run(scheduler.run_once())
        self.assertTrue(worked)
        self.assertEqual(fire.status, "PARTIAL")
        self.assertEqual(fire.completed_count, 1)
        self.assertEqual(fire.failed_count, 0)
        self.assertEqual(fire.target_count, 2)
        self.assertEqual(fire.run_count, 1)

    def test_agent_request_reuses_standard_turn_entry(self) -> None:
        runtime = AsyncMock()
        turns = AsyncMock()
        turns.start_scheduled_inspection.return_value = {
            "conversation_count": 2
        }
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=AsyncMock(),
            conversation_turn_service=turns,
        )
        fire_id = uuid7()
        payload = {
            "inspection_fire_id": str(fire_id),
            "domain_id": 200,
            "actor_id": "system:inspection-scheduler",
            "agent_id": str(uuid7()),
            "plan_display_name": "数据库日报",
            "template_id": "database_daily",
            "template_version": "1.0.0",
            "schedule_type": "DAILY",
            "timezone": "Asia/Shanghai",
            "period_start": "2026-07-22T16:00:00+00:00",
            "period_end": "2026-07-23T16:00:00+00:00",
            "timeout_seconds": 3600,
            "trace_id": "trace-inspection",
        }
        asyncio.run(
            sink.publish("OPS_INSPECTION_AGENT_REQUESTED", payload)
        )
        turns.start_scheduled_inspection.assert_awaited_once_with(payload)
        runtime.create_run.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
