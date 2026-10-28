"""Run 内人工等待的真实用例接线；不以本地替身冒充多进程或目标库验收。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch

import asyncio

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.runtime.service import _replan_budget_exhausted
from aiops_agent.domain.operations.budget import (
    ExecutionBudgetError, ensure_run_resumable, pause_task_budget, resume_task_budget,
    run_is_due, sync_run_budget, task_deadline,
)
from aiops_agent.entities import OpsRunEntity, OpsTaskEntity
from aiops_agent.repositories.runtime import OpsRunRepository
from aiops_agent.workers.task_worker import AIOpsTaskWorker
from platform_core.contracts.aiops import ArtifactInput, SuspendOpsTaskCommand
from platform_core.contracts.aiops.public import DiagnosticQueryApprovalDecision, HitlResponse
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=UTC)
IS_ORACLE = OpsRunEntity.__tablename__ == "KBOT_OPS_RUN"


def run_entity():
    run = OpsRunEntity(
        ops_run_id=uuid7(), domain_id=1 if IS_ORACLE else uuid7(), target_id=uuid7(),
        actor_id="operator" if IS_ORACLE else uuid7(), agent_id=uuid7(), agent_version_id=uuid7(),
        status="RUNNING", workflow_kind="TEST", trigger_type="CHAT", row_version=1,
        deadline_at=NOW + timedelta(seconds=100), trace_id="budget-test", created_at=NOW,
        plan_snapshot_json={}, policy_snapshot_json={},
    )
    return run


def task_entity(run, key="query", *, status="RUNNING"):
    return OpsTaskEntity(
        ops_task_id=uuid7(), ops_run_id=run.ops_run_id, task_key=key, task_type="TOOL_INVOKE",
        status=status, row_version=1, timeout_seconds=30, started_at=NOW,
        available_at=NOW, attempt_count=1, max_attempts=1, resume_pending=0,
        lease_token=uuid7(), lease_owner="worker", lease_until=NOW + timedelta(seconds=30),
        handler_id="test.query", handler_version="1", input_schema_version="TEST.v1",
        output_schema_version="TEST.v1", depends_on_json=[], input_artifacts_json=[],
    )


class ExecutionBudgetTest(TestCase):
    def test_parallel_ready_and_retry_work_do_not_pause_run(self):
        for status in ("RUNNING", "READY", "RETRY_WAIT"):
            with self.subTest(status=status):
                run = run_entity()
                waiting = task_entity(run, status="WAITING_APPROVAL")
                sync_run_budget(run, [waiting, task_entity(run, "other", status=status)], now=NOW)
                self.assertEqual(run.deadline_at, NOW + timedelta(seconds=100))
                self.assertIsNone(run.budget_paused_at)
                self.assertEqual(run.status, "RUNNING")

    def test_pause_is_idempotent_and_ignores_blocked_successors(self):
        run = run_entity()
        run.human_wait_until = NOW + timedelta(days=1)
        tasks = [task_entity(run, status="WAITING_INPUT"), task_entity(run, "answer", status="PENDING")]
        sync_run_budget(run, tasks, now=NOW + timedelta(seconds=10, microseconds=1234))
        self.assertEqual(run.budget_remaining_ms, 89998)
        sync_run_budget(run, tasks, now=NOW + timedelta(minutes=5))
        self.assertEqual(run.budget_remaining_ms, 89998)
        self.assertEqual(run.budget_paused_at, NOW + timedelta(seconds=10, microseconds=1234))
        self.assertIsNone(run.deadline_at)
        self.assertFalse(run_is_due(run, now=NOW + timedelta(hours=1)))
        self.assertTrue(run_is_due(run, now=run.human_wait_until))

    def test_mixed_waiting_phase_changes_without_reissuing_budget(self):
        run = run_entity()
        run.human_wait_until = NOW + timedelta(days=1)
        approval = task_entity(run, status="WAITING_APPROVAL")
        manual = task_entity(run, "manual", status="WAITING_INPUT")
        sync_run_budget(run, [approval, manual], now=NOW)
        self.assertEqual(run.status, "WAITING_APPROVAL")
        approval.status = "SUCCEEDED"
        sync_run_budget(run, [approval, manual], now=NOW + timedelta(minutes=1))
        self.assertEqual(run.status, "WAITING_INPUT")
        self.assertEqual(run.budget_remaining_ms, 100000)

    def test_task_and_run_resume_only_remaining_budget_and_survive_decimal_storage(self):
        run = run_entity()
        run.human_wait_until = NOW + timedelta(days=1)
        task = task_entity(run)
        pause_task_budget(task, now=NOW + timedelta(seconds=10))
        task.status = "WAITING_APPROVAL"
        sync_run_budget(run, [task], now=NOW + timedelta(seconds=10))
        task.budget_remaining_ms = Decimal(task.budget_remaining_ms)
        run.budget_remaining_ms = Decimal(run.budget_remaining_ms)
        resumed = NOW + timedelta(hours=1)
        resume_task_budget(task, now=resumed)
        task.status = "READY"
        sync_run_budget(run, [task], now=resumed)
        self.assertEqual(task_deadline(task), resumed + timedelta(seconds=20))
        self.assertEqual(run.deadline_at, resumed + timedelta(seconds=90))
        self.assertEqual(task.started_at, NOW)
        self.assertEqual(task.resume_pending, 1)
        self.assertEqual(task.attempt_count, 1)
        # 新一轮等待只扣实际执行的五秒；不会恢复为完整三十秒。
        pause_task_budget(task, now=resumed + timedelta(seconds=5))
        self.assertEqual(task.budget_remaining_ms, 15000)

    def test_retry_does_not_reset_original_task_timeout(self):
        run = run_entity()
        task = task_entity(run, status="RETRY_WAIT")
        task.attempt_count = 2
        self.assertEqual(task_deadline(task), NOW + timedelta(seconds=30))

    def test_replan_elapsed_limit_excludes_only_globally_paused_human_wait(self):
        run = run_entity()
        run.human_wait_until = NOW + timedelta(days=1)
        task = task_entity(run, status="WAITING_INPUT")
        sync_run_budget(run, [task], now=NOW + timedelta(seconds=10))
        task.status = "READY"
        sync_run_budget(run, [task], now=NOW + timedelta(seconds=610))
        self.assertEqual(run.budget_waited_ms, 600000)
        turn = SimpleNamespace(started_at=NOW, current_plan_revision=1)
        self.assertFalse(_replan_budget_exhausted(turn, now=NOW + timedelta(seconds=610),
            max_replans=2, max_seconds=180, excluded_wait_milliseconds=run.budget_waited_ms))
        self.assertTrue(_replan_budget_exhausted(turn, now=NOW + timedelta(seconds=780),
            max_replans=2, max_seconds=180, excluded_wait_milliseconds=run.budget_waited_ms))
        self.assertTrue(_replan_budget_exhausted(turn, now=NOW + timedelta(seconds=610),
            max_replans=0, max_seconds=180, excluded_wait_milliseconds=run.budget_waited_ms))

    def test_expired_zero_invalid_and_reversed_checkpoints_are_rejected(self):
        for kind in ("expired", "zero", "reversed", "cancelled", "terminal"):
            with self.subTest(kind=kind):
                run = run_entity()
                run.human_wait_until = NOW + timedelta(days=1)
                sync_run_budget(run, [task_entity(run, status="WAITING_INPUT")], now=NOW)
                now = NOW + timedelta(minutes=1)
                if kind == "expired":
                    now = run.human_wait_until
                elif kind == "zero":
                    run.budget_remaining_ms = 0
                elif kind == "reversed":
                    now = NOW - timedelta(seconds=1)
                elif kind == "cancelled":
                    run.cancel_requested_at = now
                else:
                    run.status = "COMPLETED"
                with self.assertRaises(ExecutionBudgetError):
                    ensure_run_resumable(run, now=now)

    def test_task_pause_does_not_accept_exhausted_or_reverse_resume(self):
        task = task_entity(run_entity())
        with self.assertRaises(ExecutionBudgetError):
            pause_task_budget(task, now=NOW + timedelta(seconds=30))
        pause_task_budget(task, now=NOW + timedelta(seconds=10))
        with self.assertRaises(ExecutionBudgetError):
            resume_task_budget(task, now=NOW)


class RuntimeFixture:
    """保留真实 Runtime、ORM 与请求合同，只替换事务仓储及通知 IO。"""

    def __init__(self):
        self.now = NOW + timedelta(seconds=10)
        self.run = run_entity()
        self.tasks = [task_entity(self.run)]
        self.hitls = []
        self.artifacts = []
        self.events = []
        self.runs = self
        self.changes = self
        self.platform_notifications = SimpleNamespace(emit_run_event=AsyncMock())
        self.commit = AsyncMock()
        self.service = object.__new__(AIOpsRuntimeService)
        self.service._uow_factory = lambda: self
        self.service._project_turn_interaction_status = AsyncMock()
        self.service._project_turn_terminal = AsyncMock()
        self.service._add_multi_resource_terminal_outbox = AsyncMock()
        self.service._project_deadline_partial = AsyncMock(return_value=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def database_now(self):
        return self.now

    async def get_run(self, **kwargs):
        return self.run

    get_run_scoped = get_run

    async def get_task(self, *, ops_task_id, **kwargs):
        return next((task for task in self.tasks if task.ops_task_id == ops_task_id), None)

    async def list_tasks(self, **kwargs):
        return self.tasks

    async def get_event_by_key(self, *, event_key, **kwargs):
        return next((item for item in self.events if item.event_key == event_key), None)

    async def append_event(self, **kwargs):
        item = SimpleNamespace(sequence_no=len(self.events) + 1, **kwargs)
        self.events.append(item)
        return item

    async def get_artifact_by_key(self, *, artifact_key, **kwargs):
        return next((item for item in self.artifacts if item.artifact_key == artifact_key), None)

    async def get_artifact(self, *, artifact_id):
        return next((item for item in self.artifacts if item.artifact_id == artifact_id), None)

    async def add_artifact(self, entity):
        entity.artifact_id = uuid7()
        entity.created_at = self.now
        self.artifacts.append(entity)
        return entity

    async def add_hitl(self, entity):
        entity.row_version = 1
        self.hitls.append(entity)
        return entity

    async def get_hitl_by_idempotency(self, *, idempotency_key, **kwargs):
        return next((item for item in self.hitls if item.idempotency_key == idempotency_key), None)

    async def get_hitl(self, *, hitl_id, **kwargs):
        return next((item for item in self.hitls if item.hitl_id == hitl_id), None)

    get_hitl_scoped = get_hitl

    async def answer_hitl(self, *, hitl_id, new_status, response_json, response_hash, **kwargs):
        hitl = await self.get_hitl(hitl_id=hitl_id)
        hitl.status, hitl.response_json, hitl.response_hash = new_status, response_json, response_hash
        return True

    async def list_pending_hitls(self, **kwargs):
        return [item for item in self.hitls if item.status == "PENDING"]

    async def lock_due_run(self, *, now):
        return self.run if run_is_due(self.run, now=now) else None

    async def find_expired_proposal(self, **kwargs):
        return None

    async def find_expired_hitl(self, *, now=None):
        now = now or self.now
        return next((item for item in self.hitls if item.status == "PENDING" and item.expires_at <= now), None)

    def suspend_command(self, task=None, *, approval=True):
        task = task or self.tasks[0]
        hitl_id = uuid7()
        common = {"hitl_id": str(hitl_id), "run_id": str(self.run.ops_run_id),
            "target_id": str(self.run.target_id), "expires_at": (NOW + timedelta(hours=2)).isoformat()}
        if approval:
            payload = {**common, "task_id": str(task.ops_task_id), "target_display_name": "MES",
                "purpose": "验证 SQL 计划", "sql_text": "SELECT 1", "query_sha256": "a" * 64,
                "policy_sha256": "b" * 64, "referenced_objects": ["pg_catalog.pg_class"],
                "projected_columns": ["oid"], "column_sensitivities": ["PUBLIC"], "parameters": {},
                "reason_codes": ["USER_APPROVAL_REQUIRED"], "max_rows": 10, "timeout_seconds": 5,
                "max_result_bytes": 1024}
            self.run.plan_snapshot_json = {"investigation_execution": {"dynamic_invocations": {
                item.task_key: {"validated_query": {"query_sha256": "a" * 64, "policy_sha256": "b" * 64}}
                for item in self.tasks}}}
            schema = "DIAGNOSTIC_QUERY_APPROVAL_REQUEST.v1"
            request_type = "DIAGNOSTIC_QUERY_APPROVAL"
        else:
            payload = {**common, "round_no": 1, "target_display_name": "MES", "db_type": "POSTGRESQL",
                "db_version": "16", "expected_instance_identity": {}, "instructions": ["只读查询"],
                "queries": [{"query_id": "query", "origin": "CATALOG", "purpose": "确认对象",
                    "diagnostic_question": "是否存在对象", "sql_text": "SELECT 1 AS oid", "sql_sha256": "a" * 64,
                    "expected_columns": ["oid"], "expected_types": ["integer"], "expected_shape": "ROW_SET",
                    "max_rows": 10, "timeout_hint_seconds": 5, "cost_warning": "只读",
                    "supports_if": "返回对象", "contradicts_if": "没有对象"}]}
            schema = "MANUAL_SQL_REQUEST.v1"
            request_type = "MANUAL_DIAGNOSTIC_SQL"
        return SuspendOpsTaskCommand(task_id=task.ops_task_id, worker_id="worker", lease_token=task.lease_token,
            trace_id="budget-test", hitl_id=hitl_id, request_type=request_type,
            assignee_user_id=self.run.actor_id, prompt_text="请核验", response_schema={},
            expires_at=NOW + timedelta(hours=2), idempotency_key=str(hitl_id),
            request_artifact=ArtifactInput(artifact_type=schema.split(".")[0], schema_version=schema,
                producer="test.query", producer_version="1", payload=payload, provenance={}))

    async def decide(self, hitl, decision="APPROVE"):
        return await self.service.decide_diagnostic_query(hitl_id=hitl.hitl_id, domain_id=self.run.domain_id,
            actor_id=self.run.actor_id, decision=DiagnosticQueryApprovalDecision(expected_row_version=1, decision=decision),
            idempotency_key="decision", trace_id="test")


class RuntimeWaitBudgetTest(IsolatedAsyncioTestCase):
    async def test_approval_survives_parallel_phase_and_keeps_task_budget(self):
        f = RuntimeFixture()
        f.tasks.append(task_entity(f.run, "parallel"))
        await f.service.suspend_task_for_input(f.suspend_command())
        self.assertEqual(f.run.status, "RUNNING")
        self.assertIsNone(f.run.budget_paused_at)
        f.now = NOW + timedelta(seconds=20)
        f.tasks[1].status = "SUCCEEDED"
        sync_run_budget(f.run, f.tasks, now=f.now)
        f.now = NOW + timedelta(minutes=10)
        await f.decide(f.hitls[0])
        self.assertEqual(f.run.deadline_at, f.now + timedelta(seconds=80))
        self.assertEqual(task_deadline(f.tasks[0]), f.now + timedelta(seconds=20))
        self.assertEqual(f.tasks[0].resume_pending, 1)
        self.assertEqual(f.tasks[0].max_attempts, 1)

    async def test_suspend_and_decision_replay_do_not_refresh_budget(self):
        f = RuntimeFixture()
        command = f.suspend_command()
        await f.service.suspend_task_for_input(command)
        f.now = NOW + timedelta(minutes=5)
        await f.service.suspend_task_for_input(command)
        self.assertEqual(f.run.budget_remaining_ms, 90000)
        self.assertEqual(len(f.hitls), 1)
        await f.decide(f.hitls[0])
        deadline = f.run.deadline_at
        task_limit = task_deadline(f.tasks[0])
        f.now += timedelta(seconds=1)
        await f.decide(f.hitls[0])
        self.assertEqual(f.run.deadline_at, deadline)
        self.assertEqual(task_deadline(f.tasks[0]), task_limit)

    async def test_rejecting_one_of_multiple_requests_preserves_wait(self):
        f = RuntimeFixture()
        f.tasks.append(task_entity(f.run, "second"))
        await f.service.suspend_task_for_input(f.suspend_command())
        await f.service.suspend_task_for_input(f.suspend_command(f.tasks[1], approval=False))
        until = f.run.human_wait_until
        f.now = NOW + timedelta(minutes=10)
        await f.decide(f.hitls[0], "REJECT")
        self.assertEqual(f.tasks[0].status, "SUCCEEDED")
        self.assertEqual(f.run.status, "WAITING_INPUT")
        self.assertEqual(f.run.budget_remaining_ms, 90000)
        self.assertEqual(f.run.human_wait_until, until)
        self.assertIsNone(f.run.deadline_at)

    async def test_manual_response_and_skip_release_successors_with_remaining_budget(self):
        for skip in (True, False):
            with self.subTest(skip=skip):
                f = RuntimeFixture()
                successor = task_entity(f.run, "answer", status="PENDING")
                successor.depends_on_json = ["query"]
                f.tasks.append(successor)
                await f.service.suspend_task_for_input(f.suspend_command(approval=False))
                f.now = NOW + timedelta(minutes=5)
                params = dict(hitl_id=f.hitls[0].hitl_id, domain_id=f.run.domain_id, actor_id=f.run.actor_id,
                    idempotency_key="answer", trace_id="test")
                if skip:
                    await f.service.skip_hitl(**params, expected_row_version=1)
                else:
                    response = HitlResponse(expected_row_version=1,
                        responses=[{"query_id": "query", "status": "SUCCEEDED", "raw_output": "oid\n1\n"}])
                    await f.service.respond_hitl(**params, response=response)
                self.assertEqual(successor.status, "READY")
                self.assertEqual(f.run.status, "RUNNING")
                self.assertEqual(f.run.deadline_at, f.now + timedelta(seconds=90))
                self.assertEqual(len(f.artifacts), 2)

    async def test_expiring_one_of_multiple_requests_is_not_orphaned_by_run_phase(self):
        f = RuntimeFixture()
        f.tasks.append(task_entity(f.run, "second"))
        await f.service.suspend_task_for_input(f.suspend_command())
        await f.service.suspend_task_for_input(f.suspend_command(f.tasks[1], approval=False))
        f.hitls[0].expires_at = NOW + timedelta(minutes=5)
        f.now = NOW + timedelta(minutes=10)
        self.assertTrue(await f.service.reconcile_once(trace_id="expire"))
        self.assertEqual(f.hitls[0].status, "EXPIRED")
        self.assertEqual(f.tasks[0].status, "SUCCEEDED")
        self.assertEqual(f.run.status, "WAITING_INPUT")
        self.assertEqual(f.run.budget_remaining_ms, 90000)

    async def test_cancel_closes_waiting_tasks_and_hitls_without_resurrection(self):
        f = RuntimeFixture()
        await f.service.suspend_task_for_input(f.suspend_command())
        await f.service.request_cancel(ops_run_id=f.run.ops_run_id, domain_id=f.run.domain_id,
            actor_id=f.run.actor_id, expected_row_version=1, idempotency_key="cancel", trace_id="test")
        self.assertEqual(f.run.status, "CANCELLED")
        self.assertEqual(f.tasks[0].status, "CANCELLED")
        self.assertEqual(f.hitls[0].status, "CANCELLED")
        with self.assertRaises(AIOpsApplicationError):
            await f.decide(f.hitls[0])

    async def test_global_human_wait_expiry_terminates_and_preserves_request_artifact(self):
        f = RuntimeFixture()
        await f.service.suspend_task_for_input(f.suspend_command())
        f.now = f.run.human_wait_until
        self.assertTrue(await f.service.reconcile_once(trace_id="expire"))
        self.assertEqual(f.run.status, "EXPIRED")
        self.assertEqual(f.tasks[0].status, "EXPIRED")
        self.assertEqual(f.hitls[0].status, "EXPIRED")
        self.assertEqual(len(f.artifacts), 1)

    async def test_cancelled_terminal_and_expired_run_cannot_accept_pending_approval(self):
        for state in ("cancelled", "terminal", "deadline"):
            with self.subTest(state=state):
                f = RuntimeFixture()
                await f.service.suspend_task_for_input(f.suspend_command())
                if state == "cancelled":
                    f.run.cancel_requested_at = f.now
                elif state == "terminal":
                    f.run.status = "PARTIAL"
                else:
                    f.now = f.run.human_wait_until
                with self.assertRaises(AIOpsApplicationError):
                    await f.decide(f.hitls[0])
                self.assertEqual(f.hitls[0].status, "PENDING")

    async def test_recreated_runtime_uses_persisted_budget_not_process_memory(self):
        f = RuntimeFixture()
        await f.service.suspend_task_for_input(f.suspend_command())
        f.service = object.__new__(AIOpsRuntimeService)
        f.service._uow_factory = lambda: f
        f.service._project_turn_interaction_status = AsyncMock()
        f.now = NOW + timedelta(hours=1)
        await f.decide(f.hitls[0])
        self.assertEqual(f.run.deadline_at, f.now + timedelta(seconds=90))
        self.assertEqual(task_deadline(f.tasks[0]), f.now + timedelta(seconds=20))

    async def test_terminal_does_not_accept_late_success_but_failure_bookkeeping_is_allowed(self):
        f = RuntimeFixture()
        f.run.status = "PARTIAL"
        with self.assertRaises(AIOpsApplicationError):
            f.service._ensure_lease(run=f.run, task=f.tasks[0], worker_id="worker",
                lease_token=f.tasks[0].lease_token, now=f.now)

    async def test_request_wait_limit_is_not_extended_by_later_request(self):
        f = RuntimeFixture()
        f.run.human_wait_until = NOW + timedelta(hours=1)
        with self.assertRaises(AIOpsApplicationError):
            await f.service.suspend_task_for_input(f.suspend_command())
        self.assertEqual(f.run.human_wait_until, NOW + timedelta(hours=1))
        self.assertEqual(len(f.hitls), 0)


class BudgetRepositoryTest(IsolatedAsyncioTestCase):
    async def test_approval_continuation_can_be_claimed_at_max_attempts_without_free_retry(self):
        run = run_entity()
        task = task_entity(run, status="WAITING_APPROVAL")
        run.human_wait_until = NOW + timedelta(days=1)
        pause_task_budget(task, now=NOW + timedelta(seconds=10))
        sync_run_budget(run, [task], now=NOW + timedelta(seconds=10))
        now = NOW + timedelta(hours=1)
        resume_task_budget(task, now=now)
        task.status = "READY"
        session = AsyncMock()
        session.execute.side_effect = [SimpleNamespace(all=lambda: [(task.ops_task_id, run.ops_run_id)]),
            SimpleNamespace(scalar_one_or_none=lambda: task)]
        repo = OpsRunRepository(session)
        repo.get_run = AsyncMock(return_value=run)
        claimed = await repo.claim_task(now=now, lease_owner="new-worker", lease_token=uuid7(),
            lease_until=now + timedelta(seconds=30))
        self.assertIs(claimed, task)
        self.assertEqual(task.attempt_count, 1)
        self.assertEqual(task.resume_pending, 0)
        self.assertEqual(task_deadline(task), now + timedelta(seconds=20))
        self.assertEqual(run.deadline_at, now + timedelta(seconds=90))
        self.assertEqual(task.started_at, NOW)
        self.assertIn("resume_pending", str(session.execute.call_args_list[0].args[0]))

    async def test_run_expiry_query_includes_human_wait_deadline(self):
        run = run_entity()
        run.human_wait_until = NOW + timedelta(days=1)
        sync_run_budget(run, [task_entity(run, status="WAITING_INPUT")], now=NOW)
        session = AsyncMock()
        session.execute.return_value = SimpleNamespace(scalars=lambda: [run.ops_run_id])
        repo = OpsRunRepository(session)
        repo.get_run = AsyncMock(return_value=run)
        due = await repo.lock_due_run(now=run.human_wait_until)
        self.assertIs(due, run)
        self.assertIn("human_wait_until", str(session.execute.call_args.args[0]))


class WorkerTaskBudgetTest(IsolatedAsyncioTestCase):
    def lease(self, remaining):
        run = run_entity()
        run.deadline_at = datetime.now(UTC) + timedelta(seconds=60)
        task = task_entity(run)
        task.execution_deadline_at = datetime.now(UTC) + timedelta(seconds=remaining)
        task.lease_until = run.deadline_at
        runtime = object.__new__(AIOpsRuntimeService)
        return runtime._task_lease(run, task, (), lease_token=task.lease_token)

    def worker(self, lease):
        worker = object.__new__(AIOpsTaskWorker)
        worker._worker_id = "worker"
        worker._service = SimpleNamespace(fail_task=AsyncMock())
        worker._handlers = SimpleNamespace(resolve=lambda *_: SimpleNamespace(output_schema_version=lease.output_schema_version))
        return worker

    async def test_context_freezes_effective_task_deadline_for_all_tools(self):
        lease = self.lease(2)
        context = AIOpsTaskWorker._context(lease)
        self.assertEqual(datetime.fromisoformat(context.deadline_at), lease.task_deadline_at)
        lease = lease.model_copy(update={"deadline_at": lease.task_deadline_at - timedelta(seconds=1)})
        self.assertEqual(datetime.fromisoformat(AIOpsTaskWorker._context(lease).deadline_at), lease.deadline_at)

    async def test_exhausted_task_does_not_start_handler_after_approval(self):
        lease = self.lease(-1)
        worker = self.worker(lease)
        worker._invoke_handler = AsyncMock()
        await worker._execute(lease)
        worker._invoke_handler.assert_not_awaited()
        self.assertEqual(worker._service.fail_task.call_args.args[0].error_code, "HANDLER_TIMEOUT")

    async def test_worker_timeout_uses_remaining_not_configured_full_timeout(self):
        lease = self.lease(2)
        worker = self.worker(lease)
        async def wait_for_cancellation(*args, **kwargs):
            await asyncio.Event().wait()
        worker._invoke_handler = AsyncMock(side_effect=wait_for_cancellation)
        worker._heartbeat_loop = AsyncMock(side_effect=wait_for_cancellation)
        with patch("aiops_agent.workers.task_worker.asyncio.wait", new=AsyncMock(return_value=(set(), set()))) as waiting:
            await worker._execute(lease)
        self.assertGreater(waiting.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(waiting.call_args.kwargs["timeout"], 2)
        self.assertEqual(worker._service.fail_task.call_args.args[0].error_code, "HANDLER_TIMEOUT")


class DeploymentBudgetContractTest(TestCase):
    def test_budget_columns_exist_in_canonical_and_scoped_upgrade(self):
        root = Path(__file__).resolve().parents[3]
        for entity, columns in ((OpsRunEntity, {"budget_paused_at", "budget_remaining_ms", "human_wait_until", "budget_waited_ms"}),
            (OpsTaskEntity, {"execution_deadline_at", "budget_paused_at", "budget_remaining_ms", "resume_pending"})):
            self.assertTrue(columns <= set(entity.__table__.columns.keys()))
        if IS_ORACLE:
            canonical = (root / "database/oracle/aiops_agent/002_ops_runtime.sql").read_text().upper()
            upgrade = (root / "database/oracle/operations/apply_aiops_schema_52.sql").read_text().upper()
            for column in ("BUDGET_PAUSED_AT", "BUDGET_REMAINING_MS", "HUMAN_WAIT_UNTIL", "EXECUTION_DEADLINE_AT", "RESUME_PENDING"):
                self.assertIn(column, canonical)
                self.assertIn(column, upgrade)
            self.assertIn("IF L_ACTIVE > 0", upgrade)
        else:
            upgrade = (root / "scripts/deployment/init_postgresql.py").read_text()
            self.assertIn("--align-aiops-run-wait-budget-only", upgrade)
            self.assertIn("升级前必须完成或取消全部旧运行", upgrade)
            self.assertIn("resume_pending SMALLINT NOT NULL DEFAULT 0", upgrade)


if __name__ == "__main__":
    import unittest
    unittest.main()
