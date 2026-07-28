"""步骤 9B Advisory 人工结果与只读效果验证测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.orchestration import (
    build_advisory_verification_blueprint,
)
from aiops_agent.workers.change_handlers import ActionVerificationHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.outbox_dispatcher import AIOpsDomainOutboxSink
from platform_core.identity import uuid7


class AdvisoryVerificationHandlerTest(unittest.TestCase):
    def test_absent_target_is_verified(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._context(active_rows=(), blocking_rows=())
            )
        )
        self.assertEqual(result.status, "VERIFIED")
        self.assertFalse(result.target_still_present)
        self.assertFalse(result.blocking_still_present)

    def test_remaining_blocker_is_not_achieved(self) -> None:
        result = asyncio.run(
            ActionVerificationHandler().execute(
                self._context(
                    active_rows=((1, 42, 9),),
                    blocking_rows=((1, 100, 1, 42),),
                )
            )
        )
        self.assertEqual(result.status, "NOT_ACHIEVED")
        self.assertTrue(result.target_still_present)
        self.assertTrue(result.blocking_still_present)

    def test_gap_never_becomes_success(self) -> None:
        context = self._context(active_rows=(), blocking_rows=())
        artifacts = tuple(
            item
            for item in context.input_artifacts
            if item["payload"].get("tool_id")
            != "db.session.blocking_chain"
        )
        result = asyncio.run(
            ActionVerificationHandler().execute(
                context.__class__(
                    **{**context.__dict__, "input_artifacts": artifacts}
                )
            )
        )
        self.assertEqual(result.status, "INCONCLUSIVE")
        self.assertIn(
            "VERIFICATION_EVIDENCE_MISSING", result.gap_codes
        )

    def test_blueprint_has_no_proposal_or_execute_task(self) -> None:
        blueprint = build_advisory_verification_blueprint(
            (
                "db.instance.identity",
                "db.session.active",
                "db.session.blocking_chain",
            )
        )
        self.assertEqual(blueprint.final_task_key, "verify")
        self.assertNotIn(
            "PROPOSE", {item.task_type for item in blueprint.tasks}
        )
        self.assertNotIn(
            "EXECUTE", {item.task_type for item in blueprint.tasks}
        )

    def _context(
        self,
        *,
        active_rows: tuple,
        blocking_rows: tuple,
    ) -> TaskExecutionContext:
        proposal_id = str(uuid7())
        source_run_id = str(uuid7())
        result_artifact_id = str(uuid7())
        target_id = str(uuid7())
        scope = {
            "schema_version": "ADVISORY_VERIFICATION_SCOPE.v1",
            "proposal_id": proposal_id,
            "source_run_id": source_run_id,
            "result_artifact_id": result_artifact_id,
            "action_template_id": "db.session.terminate",
            "canonical_parameters": {
                "session_id": 42,
                "serial_number": 9,
                "instance_id": 1,
            },
            "verification_tool_refs": [
                "db.session.active",
                "db.session.blocking_chain",
            ],
            "source_result_status": "EXECUTED",
        }
        return TaskExecutionContext(
            run_id=str(uuid7()),
            task_id=str(uuid7()),
            task_key="verify",
            target_id=target_id,
            agent_id=str(uuid7()),
            trigger_type="API",
            trace_id="trace-verification",
            attempt=1,
            deadline_at=None,
            plan_snapshot={"advisory_verification": scope},
            policy_snapshot={},
            input_artifacts=(
                {
                    "schema_version": (
                        "ADVISORY_VERIFICATION_SCOPE.v1"
                    ),
                    "payload": scope,
                },
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.session.active",
                    columns=(
                        "instance_id",
                        "session_id",
                        "serial_number",
                    ),
                    rows=active_rows,
                ),
                self._diagnostic(
                    target_id=target_id,
                    tool_id="db.session.blocking_chain",
                    columns=(
                        "waiting_instance_id",
                        "waiting_session_id",
                        "blocking_instance_id",
                        "blocking_session_id",
                    ),
                    rows=blocking_rows,
                ),
            ),
        )

    @staticmethod
    def _diagnostic(
        *,
        target_id: str,
        tool_id: str,
        columns: tuple[str, ...],
        rows: tuple,
    ) -> dict:
        return {
            "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
            "payload": {
                "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
                "target_id": target_id,
                "tool_id": tool_id,
                "status": "SUCCEEDED",
                "observation": {
                    "schema_version": "DATABASE_OBSERVATION.v1",
                    "executor_request_id": str(uuid7()),
                    "target_id": target_id,
                    "tool_id": tool_id,
                    "tool_version": "1.0.0",
                    "variant": "oracle_19_plus_gv",
                    "template_sha256": "1" * 64,
                    "db_type": "ORACLE",
                    "db_version": "19.0.0",
                    "capability_snapshot_hash": "2" * 64,
                    "captured_at": "2026-07-24T10:00:00Z",
                    "duration_ms": 5,
                    "columns": [
                        {
                            "name": name,
                            "logical_type": "INTEGER",
                            "sensitivity": "PUBLIC",
                        }
                        for name in columns
                    ],
                    "rows": rows,
                    "row_count": len(rows),
                    "truncated": False,
                    "result_sha256": (
                        "3" * 64
                        if tool_id == "db.session.active"
                        else "4" * 64
                    ),
                    "parameters_sha256": "5" * 64,
                },
            },
        }


class AdvisoryVerificationOutboxTest(unittest.TestCase):
    def test_executed_result_creates_idempotent_verify_run(self) -> None:
        runtime = AsyncMock()
        fallback = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=fallback,
        )
        proposal_id = str(uuid7())
        payload = {
            "proposal_id": proposal_id,
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "app_id": 100,
            "domain_id": 200,
            "actor_id": "portal:user-1",
            "agent_id": str(uuid7()),
            "target_id": str(uuid7()),
            "action_template_id": "db.session.terminate",
            "canonical_parameters": {"session_id": 42},
            "verification_tool_refs": [
                "db.session.active",
                "db.session.blocking_chain",
            ],
            "source_result_status": "EXECUTED",
            "trace_id": "trace-1",
        }
        asyncio.run(
            sink.publish("OPS_ADVISORY_RESULT_RECORDED", payload)
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual(
            command.idempotency_key,
            f"proposal:{proposal_id}:manual-result:verify",
        )
        self.assertEqual(
            command.blueprint_id, "change.advisory-verify"
        )
        fallback.publish.assert_not_awaited()

    def test_execution_result_creates_execution_scoped_verify_run(
        self,
    ) -> None:
        runtime = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime,
            fallback=AsyncMock(),
        )
        execution_id = str(uuid7())
        payload = {
            "execution_id": execution_id,
            "proposal_id": str(uuid7()),
            "source_run_id": str(uuid7()),
            "result_artifact_id": str(uuid7()),
            "app_id": 100,
            "domain_id": 200,
            "actor_id": "portal:user-1",
            "agent_id": str(uuid7()),
            "target_id": str(uuid7()),
            "trace_id": "trace-execution",
        }
        asyncio.run(
            sink.publish("OPS_EXECUTION_VERIFY_REQUESTED", payload)
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual(
            command.idempotency_key,
            f"execution:{execution_id}:verify",
        )
        self.assertEqual(
            command.client_metadata["trigger"], "execution_result"
        )


class ProposalExpiryTest(unittest.TestCase):
    def test_reconciler_expires_orphaned_hitl_without_reopening_run(
        self,
    ) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            status="EXPIRED",
        )
        task = SimpleNamespace(
            ops_task_id=uuid7(),
            ops_run_id=run.ops_run_id,
            status="EXPIRED",
        )
        hitl = SimpleNamespace(
            hitl_id=uuid7(),
            ops_task_id=task.ops_task_id,
            status="PENDING",
            expires_at=now - timedelta(seconds=1),
            responded_by=None,
            responded_at=None,
            response_json=None,
            response_hash=None,
        )
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_task=AsyncMock(return_value=task),
                get_run=AsyncMock(return_value=run),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(return_value=None),
                find_expired_hitl=AsyncMock(return_value=hitl),
                get_hitl=AsyncMock(return_value=hitl),
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsRuntimeService(
            uow_factory=lambda: context,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-orphan-hitl")
        )
        self.assertTrue(worked)
        self.assertEqual(hitl.status, "EXPIRED")
        self.assertEqual(run.status, "EXPIRED")
        self.assertEqual(task.status, "EXPIRED")
        self.assertEqual(
            hitl.response_json["reason"],
            "PARENT_STATE_NOT_WAITING_INPUT",
        )
        uow.commit.assert_awaited_once()

    def test_reconciler_expires_advisory_proposal(self) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        proposal = SimpleNamespace(
            proposal_id=uuid7(),
            ops_run_id=uuid7(),
            ops_task_id=uuid7(),
            status="ADVISORY_READY",
            expires_at=now - timedelta(seconds=1),
            updated_at=now - timedelta(minutes=1),
        )
        run = SimpleNamespace(ops_run_id=proposal.ops_run_id)
        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_run=AsyncMock(return_value=run),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(
                    return_value=proposal
                ),
                get_proposal=AsyncMock(return_value=proposal),
                get_pending_hitl=AsyncMock(return_value=None),
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        factory = lambda: context
        service = AIOpsRuntimeService(
            uow_factory=factory,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-expiry")
        )
        self.assertTrue(worked)
        self.assertEqual(proposal.status, "EXPIRED")
        uow.commit.assert_awaited_once()
        event = uow.runs.append_event.await_args.kwargs
        self.assertEqual(event["event_type"], "proposal.expired")

    def test_reconciler_turns_stale_running_into_unknown(self) -> None:
        now = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        execution = SimpleNamespace(
            execution_id=uuid7(),
            proposal_id=uuid7(),
            ops_run_id=uuid7(),
            ops_task_id=uuid7(),
            target_id=uuid7(),
            status="RUNNING",
            status_version=3,
            deadline_at=now - timedelta(seconds=1),
            action_template_id="db.session.terminate",
            executor_request_id=str(uuid7()),
            executor_instance_id="executor-test",
            grant_jti_hash="a" * 64,
            proposal_hash="b" * 64,
            command_hash="c" * 64,
            result_artifact_id=None,
            result_hash=None,
            completed_at=None,
            error_code=None,
            error_message=None,
            updated_at=now,
        )
        artifact_id = uuid7()
        run = SimpleNamespace(
            ops_run_id=execution.ops_run_id,
            actor_id="portal:user-1",
            agent_id=uuid7(),
            target_id=execution.target_id,
            plan_snapshot_json={
                "target": {
                    "app_id": 100,
                    "domain_id": 200,
                    "security_level": 3,
                }
            },
        )
        proposal = SimpleNamespace(
            proposal_id=execution.proposal_id,
            ops_task_id=execution.ops_task_id,
        )

        async def add_artifact(entity):
            entity.artifact_id = artifact_id
            return entity

        uow = SimpleNamespace(
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                lock_due_run=AsyncMock(return_value=None),
                get_run=AsyncMock(return_value=run),
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            changes=SimpleNamespace(
                find_expired_proposal=AsyncMock(return_value=None),
                find_expired_hitl=AsyncMock(return_value=None),
                find_due_execution=AsyncMock(return_value=execution),
                get_proposal=AsyncMock(return_value=proposal),
                get_execution=AsyncMock(return_value=execution),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsRuntimeService(
            uow_factory=lambda: context,
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        worked = asyncio.run(
            service.reconcile_once(trace_id="trace-reconcile")
        )
        self.assertTrue(worked)
        self.assertEqual(execution.status, "UNKNOWN")
        self.assertEqual(execution.status_version, 4)
        self.assertEqual(execution.result_artifact_id, artifact_id)
        uow.outbox.add.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
