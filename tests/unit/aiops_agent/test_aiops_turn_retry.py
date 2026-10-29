"""规划投递断点恢复、幂等与不可重试错误的跨项目回归。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from loguru import logger

from aiops_agent.application.investigation.errors import TurnPlanningStageError
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.application.turn_queue import TurnQueueService
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink, AIOpsOutboxDispatcher
from platform_core.identity import uuid7
from platform_core.prompts import PromptIntegrityError, PromptNotFoundError
from tests.unit.aiops_agent import test_aiops_turn_application as support


class _RetryOutbox:
    """模拟投递记录的状态推进，不跳过实际路由与诊断委派。"""

    def __init__(self, payload, now):
        self.message = SimpleNamespace(
            outbox_id=uuid7(), event_type="aiops.turn.understanding_requested",
            payload_json=payload, status="PENDING", available_at=now,
            attempt_count=0, max_attempts=3,
        )
        self.releases = []
        self.published = 0

    async def recover_expired(self, **_):
        return False

    async def claim(self, *, now, **_):
        if self.message.status not in {"PENDING", "RETRY_WAIT"}:
            return None
        if self.message.available_at > now:
            return None
        self.message.attempt_count += 1
        self.message.status = "PUBLISHING"
        return self.message

    async def release_failed(self, **kwargs):
        self.releases.append(kwargs)
        self.message.status = kwargs["new_status"]
        self.message.available_at = kwargs["available_at"]
        return True

    async def mark_published(self, **_):
        self.message.status = "PUBLISHED"
        self.published += 1
        return True


class TurnRetryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.uow = support._Uow()
        await support.ConversationTurnApplicationTest._start(self, self.uow)
        await TurnQueueService(uow_factory=lambda: self.uow).accept_created(
            dict(self.uow.outbox.rows[0].payload_json)
        )
        self.payload = dict(self.uow.outbox.rows[1].payload_json)
        self.turn = self.uow.turns.turns[0]
        self.planner = support._turn_planner(self.uow)
        self.planner._plan = AsyncMock(wraps=self.planner._plan)
        self.now = datetime.now(UTC)
        self.uow.runs.database_now = AsyncMock(side_effect=lambda: self.now)
        self.outbox = _RetryOutbox(self.payload, self.now)
        self.uow.outbox = self.outbox
        self.task_ids = []
        self.execute = AsyncMock()
        self.planning = object.__new__(TurnPlanningService)
        self.planning._uow_factory = lambda: self.uow
        self.planning.execute = self.execute
        self.sink = AIOpsDomainOutboxSink(
            runtime_service=object(), fallback=support._NoopSink(),
            turn_planner_service=self.planner, turn_planning_service=self.planning,
        )
        self.dispatcher = AIOpsOutboxDispatcher(
            uow_factory=lambda: self.uow, sink=self.sink,
            dispatcher_id="retry-test", lease_seconds=30, interval_seconds=1,
        )

    async def _commit_plan(self, payload):
        self.task_ids.append(uuid7())
        self.turn.current_plan_artifact_id = uuid7()
        self.turn.status = "COLLECTING"
        return {"status": "COLLECTING", "ops_run_id": payload["ops_run_id"]}

    async def _retry_once(self):
        self.now = self.outbox.message.available_at + timedelta(milliseconds=1)
        return await self.dispatcher.run_once()

    async def test_transient_failure_retries_actual_planning_on_same_run(self):
        async def execute(payload):
            if self.execute.await_count == 1:
                raise TurnPlanningStageError(RuntimeError("暂时不可用"))
            return await self._commit_plan(payload)

        self.execute.side_effect = execute
        self.assertTrue(await self.dispatcher.run_once())
        self.assertEqual("RETRY_WAIT", self.outbox.message.status)
        self.assertEqual(0, self.outbox.published)
        run = self.uow.runs.rows[0]
        original_budget = (run.started_at, run.deadline_at)
        self.assertTrue(await self._retry_once())
        self.assertEqual("PUBLISHED", self.outbox.message.status)
        self.assertEqual("COLLECTING", self.turn.status)
        self.assertEqual(2, self.execute.await_count)
        self.assertEqual(1, self.planner._plan.await_count)
        self.assertEqual(1, len(self.uow.runs.rows))
        self.assertEqual(1, len(self.uow.turns.run_links))
        self.assertEqual(1, len(self.task_ids))
        self.assertEqual(original_budget, (run.started_at, run.deadline_at))
        self.assertEqual(
            self.execute.await_args_list[0].args[0]["ops_run_id"],
            self.execute.await_args_list[1].args[0]["ops_run_id"],
        )

    async def test_repeated_failure_reaches_exhaustion_not_false_success(self):
        self.execute.side_effect = TurnPlanningStageError(RuntimeError("暂时不可用"))
        await self.dispatcher.run_once()
        await self._retry_once()
        self.assertEqual("RETRY_WAIT", self.outbox.message.status)
        self.assertEqual(2, self.execute.await_count)
        await self._retry_once()
        self.assertEqual("FAILED", self.outbox.message.status)
        self.assertEqual(0, self.outbox.published)
        self.assertEqual(3, self.execute.await_count)
        self.assertEqual("FAILED", self.turn.status)
        self.assertEqual("FAILED", self.uow.runs.rows[0].status)
        self.assertEqual("AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR", self.turn.error_code)
        self.assertEqual(self.turn.error_code, self.uow.runs.rows[0].error_code)
        self.assertEqual([], self.task_ids)

    async def test_plan_committed_before_ack_is_not_executed_again(self):
        async def execute(payload):
            await self._commit_plan(payload)
            raise TurnPlanningStageError(RuntimeError("提交后投递确认中断"))

        self.execute.side_effect = execute
        await self.dispatcher.run_once()
        await self._retry_once()
        self.assertEqual("PUBLISHED", self.outbox.message.status)
        self.assertEqual(1, self.execute.await_count)
        self.assertEqual(1, len(self.task_ids))
        self.assertEqual(1, len(self.uow.runs.rows))

    async def test_routing_commit_is_replayed_without_rerouting(self):
        await self.planner.begin(self.payload)
        self.execute.side_effect = self._commit_plan
        await self.dispatcher.run_once()
        self.assertEqual(1, self.planner._plan.await_count)
        self.assertEqual(1, self.execute.await_count)
        self.assertEqual("COLLECTING", self.turn.status)

    async def test_clarification_commit_replays_pending_diagnosis(self):
        await self.planner.begin(self.payload)
        payload = {**self.payload, "response": {"candidate_id": "resolved"}}
        self.execute.side_effect = self._commit_plan
        await self.sink.publish("aiops.turn.clarification_resolved", payload)
        self.assertEqual(1, self.execute.await_count)
        self.assertEqual(1, self.planner._plan.await_count)

    async def test_concurrent_route_commit_returns_continuation(self):
        await self.planner.begin(self.payload)
        result = await self.planner._persist_route(
            self.payload, {"turn_id": str(self.turn.turn_id)},
            None, None, None, None, None,
        )
        self.assertEqual("ACTIVE_DIAGNOSIS", result["dispatch"])
        self.assertEqual([], self.uow.runs.rows)

    async def test_planning_checkpoint_can_be_resumed(self):
        await self.planner.begin(self.payload)
        await self.planner.start_active_diagnosis(self.payload)
        self.turn.status = "PLANNING"
        self.execute.side_effect = self._commit_plan
        await self.dispatcher.run_once()
        self.assertEqual(1, self.execute.await_count)
        self.assertEqual(1, len(self.uow.runs.rows))

    async def test_terminal_waiting_and_execution_states_do_not_restart(self):
        await self.planner.begin(self.payload)
        for status in (
            "COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "WAITING_USER",
            "WAITING_INPUT", "WAITING_APPROVAL", "COLLECTING", "ANSWERING", "REPLANNING",
        ):
            with self.subTest(status=status):
                self.turn.status = status
                await self.sink.publish("aiops.turn.understanding_requested", self.payload)
                self.assertEqual(status, self.turn.status)
        self.execute.assert_not_awaited()
        self.assertEqual([], self.uow.runs.rows)
        self.assertEqual(1, self.planner._plan.await_count)

    async def test_cancel_requested_does_not_restart(self):
        await self.planner.begin(self.payload)
        self.turn.cancel_requested_at = self.now
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        self.execute.assert_not_awaited()
        self.assertEqual([], self.uow.runs.rows)

    async def test_existing_run_terminal_cancelled_paused_or_due_does_not_restart(self):
        await self.planner.begin(self.payload)
        await self.planner.start_active_diagnosis(self.payload)
        run = self.uow.runs.rows[0]
        deadline = run.deadline_at
        for status in ("COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"):
            with self.subTest(status=status):
                run.status = status
                await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        run.status = "RUNNING"
        run.cancel_requested_at = self.now
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        run.cancel_requested_at = None
        run.budget_paused_at = self.now
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        run.budget_paused_at = None
        run.deadline_at = self.now - timedelta(seconds=1)
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        self.execute.assert_not_awaited()
        self.assertEqual(1, len(self.uow.runs.rows))
        self.assertNotEqual(deadline, run.deadline_at)

    async def test_expired_unstarted_route_does_not_create_run(self):
        await self.planner.begin(self.payload)
        self.turn.orchestration_internal_json = {
            **self.turn.orchestration_internal_json,
            "planning_deadline_at": (self.now - timedelta(seconds=1)).isoformat(),
        }
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        self.execute.assert_not_awaited()
        self.assertEqual("FAILED", self.turn.status)
        self.assertEqual("TURN_DEADLINE_EXCEEDED", self.turn.error_code)
        self.assertEqual([], self.uow.runs.rows)

    async def test_multi_resource_launch_recovery_only_while_incomplete(self):
        await self.planner.begin(self.payload)
        self.turn.status = "COLLECTING"
        self.turn.orchestration_internal_json = {
            **self.turn.orchestration_internal_json,
            "multi_resource_diagnosis": {"launch_complete": False},
        }
        resumed = await self.planner.begin(self.payload)
        self.assertEqual("ACTIVE_DIAGNOSIS", resumed["dispatch"])
        self.turn.orchestration_internal_json["multi_resource_diagnosis"]["launch_complete"] = True
        completed = await self.planner.begin(self.payload)
        self.assertIsNone(completed["dispatch"])
        self.assertEqual(1, self.planner._plan.await_count)

    async def test_prompt_configuration_error_fails_turn_and_run_immediately(self):
        for cause, code, detail in (
            (PromptNotFoundError("不得输出的内部正文"), "AIOPS_PROMPT_NOT_FOUND", "prompt-not-found"),
            (PromptIntegrityError("不得输出的内部正文"), "AIOPS_PROMPT_INTEGRITY_ERROR", "prompt-integrity-violation"),
        ):
            with self.subTest(code=code):
                error = TurnPlanningStageError(cause)
                self.assertFalse(error.retryable)
                self.assertEqual(code, error.code)
                self.assertEqual(detail, error.safe_detail)
                self.assertNotIn("不得输出", str(error))
        self.execute.side_effect = TurnPlanningStageError(PromptNotFoundError("不得输出的内部正文"))
        await self.dispatcher.run_once()
        self.assertEqual(1, self.execute.await_count)
        self.assertEqual("FAILED", self.outbox.message.status)
        self.assertEqual("AIOPS_PROMPT_NOT_FOUND", self.outbox.releases[0]["error_code"])
        self.assertEqual("AIOPS_PROMPT_NOT_FOUND", self.turn.error_code)
        self.assertEqual("FAILED", self.uow.runs.rows[0].status)
        self.assertIn("Prompt", self.turn.error_message)
        self.assertNotIn("不得输出", self.turn.error_message)

    async def test_partial_multi_resource_launch_retries_without_duplicate_runs(self):
        await self.planner.begin(self.payload)
        internal = self.turn.orchestration_internal_json
        first = internal["selected_scope"]["execution_targets"][0]
        internal["selected_scope"]["execution_targets"] = [
            first,
            {**first, "resource_id": str(uuid7()), "target_id": str(uuid7())},
        ]
        runtime = support._MultiResourceRuntime(self.uow)
        create_run = runtime.create_run

        async def interrupted_create(command):
            if create.await_count == 2:
                raise RuntimeError("第二个资源创建前连接中断")
            return await create_run(command)

        create = AsyncMock(side_effect=interrupted_create)
        runtime.create_run = create
        self.planner._runtime_service = runtime
        await self.dispatcher.run_once()
        self.assertEqual("RETRY_WAIT", self.outbox.message.status)
        self.assertEqual(1, len(self.uow.runs.rows))
        self.assertFalse(
            self.turn.orchestration_internal_json["multi_resource_diagnosis"]["launch_complete"]
        )
        original_run_id = self.uow.runs.rows[0].ops_run_id

        await self._retry_once()
        self.assertEqual("PUBLISHED", self.outbox.message.status)
        self.assertEqual(2, len(self.uow.runs.rows))
        self.assertEqual(2, len(self.uow.turns.run_links))
        self.assertEqual(2, len(runtime.commands))
        self.assertEqual(original_run_id, self.uow.runs.rows[0].ops_run_id)
        coordination = self.turn.orchestration_internal_json["multi_resource_diagnosis"]
        self.assertTrue(coordination["launch_complete"])
        self.assertEqual(2, len(coordination["ops_run_ids"]))
        self.assertEqual(1, self.planner._plan.await_count)
        self.execute.assert_not_awaited()
        await self.sink.publish("aiops.turn.understanding_requested", self.payload)
        self.assertEqual(4, create.await_count)

    async def test_planning_retry_logs_keep_safe_failure_stage(self):
        async def execute(_):
            try:
                raise PromptIntegrityError("不得输出的内部正文")
            except PromptIntegrityError as cause:
                raise TurnPlanningStageError(cause) from cause

        self.execute.side_effect = execute
        messages = []
        sink_id = logger.add(messages.append, format="{message}", level="WARNING")
        try:
            await self.dispatcher.run_once()
        finally:
            logger.remove(sink_id)
        logged = "\n".join(messages)
        self.assertIn("code=AIOPS_PROMPT_INTEGRITY_ERROR", logged)
        self.assertIn("cause=PromptIntegrityError", logged)
        self.assertIn("detail=prompt-integrity-violation", logged)
        self.assertNotIn("不得输出", logged)


if __name__ == "__main__":
    unittest.main()
