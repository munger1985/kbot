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
            "manual_result_status": "EXECUTED",
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
            "manual_result_status": "EXECUTED",
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


class ProposalExpiryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
