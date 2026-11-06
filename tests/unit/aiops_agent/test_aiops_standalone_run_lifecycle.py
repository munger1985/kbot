"""独立运行不得阻塞对话任务领取及期限收尾的回归。"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.dialects import oracle, postgresql

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.repositories.runtime import OpsRunRepository
from platform_core.contracts.aiops import ClaimOpsTaskCommand
from platform_core.identity import uuid7


NOW = datetime(2026, 11, 6, 0, 50, 20, tzinfo=timezone.utc)


def fixture(workflow="INSPECTION", *, linked=False):
    turn_id = uuid7()
    turn = SimpleNamespace(turn_id=turn_id, status="ANSWERING")
    run = SimpleNamespace(
        ops_run_id=uuid7(), domain_id=1, workflow_kind=workflow,
        status="CREATED", started_at=None, completed_at=None,
        deadline_at=NOW + timedelta(seconds=30), plan_snapshot_json={},
    )
    if linked:
        run.plan_snapshot_json = {"turn_id": str(turn_id), "conversation_id": str(uuid7())}
    task = SimpleNamespace(
        ops_task_id=uuid7(), ops_run_id=run.ops_run_id,
        task_key="answer:compose" if linked else "scope", attempt_count=1,
        task_type="CONTEXT_BUILD", status="READY", started_at=NOW,
        timeout_seconds=30, completed_at=None, lease_until=None,
    )
    turns = SimpleNamespace(
        get_run_link_by_ops_run_id=AsyncMock(
            return_value=SimpleNamespace(turn_id=turn_id) if linked else None
        ),
        get_turn=AsyncMock(return_value=turn),
        get_tool_invocation_by_task=AsyncMock(return_value=None),
    )
    runs = SimpleNamespace(
        database_now=AsyncMock(return_value=NOW),
        get_run=AsyncMock(return_value=run),
        claim_task=AsyncMock(return_value=task),
        append_event=AsyncMock(return_value=SimpleNamespace(sequence_no=1)),
        list_tasks=AsyncMock(return_value=[task]),
        lock_due_run=AsyncMock(return_value=run),
    )

    class Uow:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

    uow = Uow()
    uow.runs, uow.turns = runs, turns
    uow.changes = SimpleNamespace(list_pending_hitls=AsyncMock(return_value=[]))
    uow.commit = AsyncMock()
    runtime = object.__new__(AIOpsRuntimeService)
    runtime._uow_factory = lambda: uow
    runtime._append_turn_event = AsyncMock()
    return runtime, uow, run, task, turn


@pytest.mark.parametrize("workflow", ["INSPECTION", "ALERT_DIAGNOSIS", "CHAT_TURN"])
def test_standalone_run_skips_all_conversation_projections(workflow):
    async def execute():
        runtime, uow, run, task, _ = fixture(workflow)
        await runtime._project_turn_task_started(uow=uow, run=run, task=task)
        await runtime._project_turn_task(
            uow=uow, run=run, task=task, artifact=SimpleNamespace(), now=NOW
        )
        for status in ("FAILED", "CANCELLED"):
            await runtime._project_turn_terminal(
                uow=uow, run=run, status=status, error_code=None,
                public_summary="运行已结束", now=NOW,
            )
        await runtime._project_turn_interaction_status(
            uow=uow, run=run, status="WAITING_USER",
            event_type="run.budget_changed", payload={},
        )
        uow.turns.get_turn.assert_not_awaited()
        runtime._append_turn_event.assert_not_awaited()
    asyncio.run(execute())


@pytest.mark.parametrize("workflow", ["INSPECTION", "ALERT_DIAGNOSIS", "CHAT_TURN"])
def test_linked_runs_keep_progress_and_terminal_projection(workflow):
    async def execute():
        runtime, uow, run, task, turn = fixture(workflow, linked=True)
        await runtime._project_turn_task_started(uow=uow, run=run, task=task)
        assert runtime._append_turn_event.await_args.kwargs["event_type"] == "thinking.delta"
        await runtime._project_turn_terminal(
            uow=uow, run=run, status="FAILED", error_code="OPS_RUN_EXPIRED",
            public_summary="诊断运行超过截止时间", now=NOW,
        )
        assert turn.status == "FAILED"
        assert turn.completed_at == NOW
        assert turn.error_code == "OPS_RUN_EXPIRED"
        assert runtime._append_turn_event.await_count == 2
    asyncio.run(execute())


@pytest.mark.parametrize("owner_key", ["turn_id", "conversation_id"])
def test_declared_conversation_owner_cannot_silently_lose_link(owner_key):
    async def execute():
        runtime, uow, run, _, _ = fixture()
        run.plan_snapshot_json = {owner_key: str(uuid7())}
        with pytest.raises(AIOpsApplicationError, match="Primary Turn"):
            await runtime._turn_for_projection(uow=uow, run=run)
    asyncio.run(execute())


def test_existing_link_cannot_reference_missing_turn():
    async def execute():
        runtime, uow, run, _, _ = fixture(linked=True)
        uow.turns.get_turn.return_value = None
        with pytest.raises(AIOpsApplicationError, match="有效 Turn"):
            await runtime._turn_for_projection(uow=uow, run=run)
    asyncio.run(execute())


def test_worker_can_claim_inspection_then_conversation_without_rollback():
    async def execute():
        runtime, uow, run, task, _ = fixture()
        runtime._input_artifacts = AsyncMock(return_value=[])
        runtime._task_lease = Mock(return_value="leased")
        claim = ClaimOpsTaskCommand(
            worker_id="worker-test", lease_seconds=30, trace_id="claim-test"
        )
        assert await runtime.claim_task(claim) == "leased"
        uow.commit.assert_awaited_once()
        assert run.status == "RUNNING"
        uow.turns.get_turn.assert_not_awaited()

        _, linked_uow, linked_run, linked_task, _ = fixture("CHAT_TURN", linked=True)
        uow.runs.get_run.return_value = linked_run
        uow.runs.claim_task.return_value = linked_task
        uow.turns = linked_uow.turns
        assert await runtime.claim_task(claim) == "leased"
        assert uow.commit.await_count == 2
        runtime._append_turn_event.assert_awaited_once()
    asyncio.run(execute())


def test_expired_inspections_no_longer_prevent_conversation_terminal():
    async def execute():
        runtime, uow, run, task, _ = fixture()
        runtime._project_deadline_partial = AsyncMock(return_value=False)
        for _ in range(2):
            run.status, task.status = "CREATED", "READY"
            assert await runtime.reconcile_once(trace_id="expire-inspection")
            assert run.status == "EXPIRED"
            assert task.status == "EXPIRED"
        _, linked_uow, linked_run, linked_task, turn = fixture("CHAT_TURN", linked=True)
        uow.runs.lock_due_run.return_value = linked_run
        uow.runs.list_tasks.return_value = [linked_task]
        uow.turns = linked_uow.turns
        assert await runtime.reconcile_once(trace_id="expire-conversation")
        assert linked_run.status == "EXPIRED"
        assert turn.status == "FAILED"
        assert turn.error_code == "OPS_RUN_EXPIRED"
        assert uow.commit.await_count == 3
    asyncio.run(execute())


@pytest.mark.parametrize("dialect", [postgresql.dialect(), oracle.dialect()])
def test_deadline_query_prioritizes_interactive_runs_without_skipping_background(dialect):
    async def execute():
        session = SimpleNamespace(execute=AsyncMock(
            return_value=SimpleNamespace(scalars=lambda: [])
        ))
        repository = OpsRunRepository(session)
        assert await repository.lock_due_run(now=NOW) is None
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
        ordering = sql.split("ORDER BY", 1)[1]
        assert ordering.index("interaction_mode") < ordering.index("deadline_at")
        assert "'INTERACTIVE'" in ordering
        assert "ELSE 1" in ordering
        assert "deadline_at" in sql.split("ORDER BY", 1)[0]
    asyncio.run(execute())
