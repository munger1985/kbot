"""Step 11A AIOps Root Delegation 安全边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.contracts.aiops import RootDelegationRequest
from platform_core.contracts.aiops.internal import OpsRunReceipt
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


class AIOpsDelegationBoundaryTest(unittest.TestCase):
    def test_create_delegation_freezes_parent_link_and_stable_key(
        self,
    ) -> None:
        delegation_id = uuid7()
        parent_run_id = uuid7()
        child_run_id = uuid7()
        request = RootDelegationRequest(
            delegation_id=delegation_id,
            parent_agent_run_id=parent_run_id,
            agent_id=uuid7(),
            target_id=uuid7(),
            domain_id="200",
            user_intent="分析数据库阻塞问题",
            deadline=datetime.now(UTC) + timedelta(minutes=10),
        )
        service = AIOpsRuntimeService(
            uow_factory=Mock(),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        service.create_run = AsyncMock(
            return_value=OpsRunReceipt(
                ops_run_id=child_run_id,
                status="CREATED",
                row_version=1,
                event_cursor=1,
            )
        )
        result = asyncio.run(
            service.create_delegated_run(
                request=request,
                domain_id=200,
                actor_id="user-1",
                trace_id="trace-1",
            )
        )
        command = service.create_run.await_args.args[0]
        self.assertEqual(
            command.idempotency_key, f"delegation:{delegation_id}"
        )
        self.assertEqual(command.parent_agent_run_id, parent_run_id)
        self.assertEqual(command.parent_delegation_id, delegation_id)
        self.assertEqual(command.blueprint_id, "diagnosis.root-cause")
        self.assertEqual(result.ops_run_id, child_run_id)

    def test_delegation_events_drop_command_and_raw_result(self) -> None:
        delegation_id = uuid7()
        run_id = uuid7()
        now = datetime.now(UTC)
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="WAITING_APPROVAL",
            trace_id="trace-child",
        )
        event = SimpleNamespace(
            sequence_no=3,
            event_type="proposal.pending_approval",
            payload_json={
                "status": "PENDING_APPROVAL",
                "proposal_id": str(uuid7()),
                "risk_level": "HIGH",
                "expires_at": now.isoformat(),
                "rendered_command": "DROP SECRET",
                "raw_sql_result": {"password": "secret"},
                "trace_id": "trace-child",
            },
            created_at=now,
        )
        runs = SimpleNamespace(
            get_by_parent_delegation_scoped=AsyncMock(
                return_value=run
            ),
            latest_event_sequence=AsyncMock(return_value=3),
            list_events_after=AsyncMock(return_value=[event]),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(SimpleNamespace(runs=runs)),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        result = asyncio.run(
            service.list_delegation_events(
                delegation_id=delegation_id,
                domain_id=200,
                after_sequence=0,
                limit=100,
            )
        )
        projected = result.events[0].model_dump(mode="json")
        self.assertEqual(projected["event_type"], "approval.required")
        self.assertEqual(projected["risk_level"], "HIGH")
        self.assertNotIn("rendered_command", projected)
        self.assertNotIn("raw_sql_result", projected)
        scoped = (
            runs.get_by_parent_delegation_scoped.await_args.kwargs
        )
        self.assertEqual(scoped["domain_id"], 200)

    def test_terminal_result_exposes_only_safe_summary_and_ref(
        self,
    ) -> None:
        delegation_id = uuid7()
        run_id = uuid7()
        artifact_id = uuid7()
        supporting_id = "a" * 64
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="COMPLETED",
            root_cause_level="PROBABLE",
            final_artifact_id=artifact_id,
            error_code=None,
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id,
            artifact_type="DIAGNOSIS_REPORT",
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            content_hash="b" * 64,
            payload_json={
                "root_cause": {
                    "effective_level": "PROBABLE",
                    "supporting_fact_refs": [supporting_id],
                },
                "facts": [
                    {
                        "fact_id": supporting_id,
                        "fact_summary": "数据库存在持续阻塞链",
                        "raw_rows": [{"secret": "不得输出"}],
                    }
                ],
                "gaps": ["MONITOR_DATA_MISSING"],
                "rendered_command": "不得输出",
            },
        )
        runs = SimpleNamespace(
            get_by_parent_delegation_scoped=AsyncMock(
                return_value=run
            ),
            get_artifact=AsyncMock(return_value=artifact),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(SimpleNamespace(runs=runs)),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        result = asyncio.run(
            service.get_delegation_result(
                delegation_id=delegation_id,
                domain_id=200,
            )
        )
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(
            result.diagnosis.artifact.artifact_id, artifact_id
        )
        self.assertIn("数据库存在持续阻塞链", result.safe_summary)
        self.assertNotIn("不得输出", result.safe_summary)


if __name__ == "__main__":
    unittest.main()
