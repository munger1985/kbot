"""步骤 9C 逐命令审批与一次性授权测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.actions import ActionRegistry, ActionRenderer
from aiops_agent.actions.grants import MutationGrantCodec
from aiops_agent.application.changes import AIOpsChangeService
from aiops_agent.application.errors import AIOpsApplicationError
from platform_core.contracts.aiops import ApprovalCommand
from platform_core.contracts.aiops.executor import (
    ExecutionStatusEvent,
    MutationClaimRequest,
)
from platform_core.identity import uuid7


class ApprovalServiceTest(unittest.TestCase):
    def test_chat_proposal_block_tracks_approval_status(self) -> None:
        proposal_id = uuid7()
        run_id = uuid7()
        block = SimpleNamespace(
            block_type="PROPOSAL_SUMMARY",
            payload_json={
                "proposal_id": str(proposal_id),
                "status": "PENDING_APPROVAL",
                "row_version": 1,
            },
            content_hash="a" * 64,
        )
        turns = SimpleNamespace(
            get_run_link_by_ops_run_id=AsyncMock(
                return_value=SimpleNamespace(turn_id=uuid7())
            ),
            list_answer_blocks=AsyncMock(return_value=[block]),
        )
        asyncio.run(
            AIOpsChangeService._update_chat_proposal_block(
                uow=SimpleNamespace(turns=turns),
                proposal=SimpleNamespace(
                    proposal_id=proposal_id,
                    ops_run_id=run_id,
                    row_version=1,
                ),
                status="APPROVED",
            )
        )
        self.assertEqual(block.payload_json["status"], "APPROVED")
        self.assertEqual(block.payload_json["row_version"], 2)
        self.assertNotEqual(block.content_hash, "a" * 64)

    def test_closed_kill_switch_never_opens_transaction(self) -> None:
        factory = unittest.mock.Mock()
        service = AIOpsChangeService(
            uow_factory=factory,
            action_registry=ActionRegistry.load(),
            approval_enabled=True,
            mutation_enabled=False,
        )
        with self.assertRaises(AIOpsApplicationError) as caught:
            asyncio.run(
                service.approve_proposal(
                    proposal_id=uuid7(),
                    domain_id=200,
                    actor_id="portal:user-1",
                    command=ApprovalCommand(
                        expected_row_version=1,
                        expected_proposal_hash="a" * 64,
                    ),
                    idempotency_key="approve-1",
                    trace_id="trace-1",
                )
            )
        self.assertEqual(caught.exception.code, "OPS_STATE_CONFLICT")
        factory.assert_not_called()

    def test_approval_atomically_creates_token_execution_and_outbox(
        self,
    ) -> None:
        now = datetime(2026, 7, 24, 11, 0, tzinfo=UTC)
        registry = ActionRegistry.load()
        template = registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="ORACLE",
            db_version="19.0.0",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        parameters = {
            "session_id": 42,
            "serial_number": 9,
            "instance_id": 1,
        }
        rendered = ActionRenderer().render(template, parameters)
        proposal_id = uuid7()
        run_id = uuid7()
        task_id = uuid7()
        target_id = uuid7()
        agent_id = uuid7()
        proposal = SimpleNamespace(
            proposal_id=proposal_id,
            ops_run_id=run_id,
            ops_task_id=task_id,
            target_id=target_id,
            row_version=1,
            proposal_hash="a" * 64,
            status="PENDING_APPROVAL",
            expires_at=now + timedelta(minutes=10),
            action_type="AGENT_EXECUTE",
            execution_mode="EXECUTABLE_AFTER_APPROVAL",
            effect_class="SESSION_CONTROL",
            executor_kind="DATABASE",
            action_template_id="db.session.terminate",
            action_template_version="1.0.0",
            action_template_hash=rendered.template_hash,
            renderer_version=rendered.renderer_version,
            command_hash=rendered.command_hash,
            parameters_json=parameters,
            parameters_hash=rendered.parameters_hash,
            policy_decision_hash="b" * 64,
            updated_at=now,
        )
        run = SimpleNamespace(
            ops_run_id=run_id,
            agent_id=agent_id,
            actor_id="portal:user-1",
            policy_snapshot_json={"policy_hash": "c" * 64},
        )
        hitl = SimpleNamespace(
            hitl_id=uuid7(),
            proposal_id=proposal_id,
            assignee_user_id="portal:user-1",
            status="PENDING",
            responded_by=None,
            responded_at=None,
            response_json=None,
            response_hash=None,
        )
        target = SimpleNamespace(
            target_id=target_id,
            status="ENABLED",
            connectivity_status="CONNECTED",
            domain_id=200,
            execution_credential_id=uuid7(),
            row_version=3,
            db_type="ORACLE",
            version_code="19.0.0",
            capabilities_json={"session_management": True},
            environment="PROD",
            security_level=3,
        )
        binding = SimpleNamespace(
            status="ACTIVE",
            allow_mutation=True,
            allowed_actions_json=["db.session.terminate"],
            policy_id=uuid7(),
        )
        policy = SimpleNamespace(
            status="ACTIVE",
            policy_hash="c" * 64,
            rules_json={
                "allow_agent_execution": True,
                "entitlements": [],
            },
        )
        changes = SimpleNamespace(
            get_execution_by_idempotency=AsyncMock(
                return_value=None
            ),
            get_proposal_scoped=AsyncMock(return_value=proposal),
            get_proposal=AsyncMock(return_value=proposal),
            get_pending_hitl=AsyncMock(return_value=hitl),
            add_approval_token=AsyncMock(
                side_effect=lambda entity: entity
            ),
            add_execution=AsyncMock(side_effect=lambda entity: entity),
        )
        uow = SimpleNamespace(
            changes=changes,
            runs=SimpleNamespace(
                get_run=AsyncMock(return_value=run),
                database_now=AsyncMock(return_value=now),
                add_artifact=AsyncMock(
                    side_effect=lambda entity: entity
                ),
                append_event=AsyncMock(),
            ),
            targets=SimpleNamespace(
                get_scoped=AsyncMock(return_value=target),
                get_agent_binding=AsyncMock(return_value=binding),
            ),
            policies=SimpleNamespace(
                get_scoped=AsyncMock(return_value=policy)
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_proposal_event=AsyncMock()
            ),
            turns=SimpleNamespace(
                get_run_link_by_ops_run_id=AsyncMock(return_value=None),
                list_answer_blocks=AsyncMock(return_value=[]),
            ),
            commit=AsyncMock(),
            rollback=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsChangeService(
            uow_factory=lambda: context,
            action_registry=registry,
            approval_enabled=True,
            mutation_enabled=True,
        )
        service._snapshot = AsyncMock(  # type: ignore[method-assign]
            return_value=SimpleNamespace(target_version=3)
        )
        receipt = asyncio.run(
            service.approve_proposal(
                proposal_id=proposal_id,
                domain_id=200,
                actor_id="portal:user-1",
                command=ApprovalCommand(
                    expected_row_version=1,
                    expected_proposal_hash="a" * 64,
                    note="已确认终止阻塞会话",
                ),
                idempotency_key="approve-proposal-1",
                trace_id="trace-approval",
            )
        )
        self.assertEqual(receipt.proposal_status, "APPROVED")
        self.assertEqual(receipt.execution_status, "CREATED")
        self.assertEqual(proposal.status, "APPROVED")
        self.assertEqual(hitl.status, "APPROVED")
        changes.add_approval_token.assert_awaited_once()
        changes.add_execution.assert_awaited_once()
        uow.outbox.add.assert_awaited_once()
        uow.commit.assert_awaited_once()
        execution = changes.add_execution.await_args.args[0]
        self.assertEqual(execution.status, "CREATED")
        self.assertEqual(execution.proposal_hash, proposal.proposal_hash)
        token = changes.add_approval_token.await_args.args[0]
        self.assertNotEqual(token.token_hash, token.nonce_hash)


class ExecutionClaimTest(unittest.TestCase):
    def test_claim_consumes_authorization_and_replays_same_grant(
        self,
    ) -> None:
        now = datetime.now(UTC)
        registry = ActionRegistry.load()
        template = registry.resolve(
            action_template_id="db.session.terminate",
            version="1.0.0",
            db_type="MYSQL",
            db_version="8.0.36",
            capabilities={"session_management"},
            entitlements=set(),
            environment="PROD",
        )
        parameters = {"session_id": 42}
        rendered = ActionRenderer().render(template, parameters)
        execution_id = uuid7()
        request_id = uuid7()
        proposal_id = uuid7()
        run_id = uuid7()
        task_id = uuid7()
        target_id = uuid7()
        agent_id = uuid7()
        token = SimpleNamespace(
            approval_token_id=uuid7(),
            hitl_id=uuid7(),
            status="ISSUED",
            expires_at=now + timedelta(minutes=5),
            target_version=7,
            policy_decision_hash="b" * 64,
            parameters_hash=rendered.parameters_hash,
            token_hash="d" * 64,
            approver_id="portal:user-1",
            consumed_at=None,
        )
        execution = SimpleNamespace(
            execution_id=execution_id,
            executor_request_id=str(request_id),
            ops_run_id=run_id,
            proposal_id=proposal_id,
            approval_token_id=token.approval_token_id,
            target_id=target_id,
            status="CREATED",
            executor_instance_id=None,
            claimed_at=None,
            grant_jti_hash=None,
            status_version=1,
            action_template_hash=rendered.template_hash,
            parameters_hash=rendered.parameters_hash,
            command_hash=rendered.command_hash,
            proposal_hash="a" * 64,
            updated_at=now,
        )
        proposal = SimpleNamespace(
            proposal_id=proposal_id,
            ops_task_id=task_id,
            status="APPROVED",
            execution_mode="EXECUTABLE_AFTER_APPROVAL",
            effect_class="SESSION_CONTROL",
            executor_kind="DATABASE",
            action_template_id="db.session.terminate",
            action_template_version="1.0.0",
            action_template_hash=rendered.template_hash,
            parameters_json=parameters,
            parameters_hash=rendered.parameters_hash,
            command_hash=rendered.command_hash,
            proposal_hash="a" * 64,
            policy_decision_hash="b" * 64,
            updated_at=now,
        )
        hitl = SimpleNamespace(status="APPROVED")
        run = SimpleNamespace(
            ops_run_id=run_id,
            agent_id=agent_id,
            trace_id="trace-source",
            plan_snapshot_json={
                "target": {"domain_id": 200}
            },
            policy_snapshot_json={"policy_hash": "c" * 64},
        )
        target = SimpleNamespace(
            target_id=target_id,
            status="ENABLED",
            connectivity_status="CONNECTED",
            domain_id=200,
            execution_credential_id=uuid7(),
            row_version=7,
            db_type="MYSQL",
            version_code="8.0.36",
            capabilities_json={"session_management": True},
            environment="PROD",
            endpoint_json={
                "host": "mysql.internal",
                "port": 3306,
                "database": "ops",
                "tls_enabled": True,
            },
        )
        binding = SimpleNamespace(
            status="ACTIVE",
            allow_mutation=True,
            allowed_actions_json=["db.session.terminate"],
            policy_id=uuid7(),
        )
        policy = SimpleNamespace(
            status="ACTIVE",
            policy_hash="c" * 64,
            rules_json={
                "allow_agent_execution": True,
                "entitlements": [],
            },
        )
        changes = SimpleNamespace(
            get_execution=AsyncMock(
                side_effect=lambda **kwargs: execution
            ),
            get_proposal=AsyncMock(return_value=proposal),
            get_approval_token=AsyncMock(return_value=token),
            get_hitl=AsyncMock(return_value=hitl),
            get_active_execution_for_target=AsyncMock(
                return_value=None
            ),
        )
        uow = SimpleNamespace(
            changes=changes,
            runs=SimpleNamespace(
                get_run=AsyncMock(return_value=run),
                database_now=AsyncMock(return_value=now),
                append_event=AsyncMock(),
            ),
            targets=SimpleNamespace(
                get_scoped=AsyncMock(return_value=target),
                get_agent_binding=AsyncMock(return_value=binding),
            ),
            policies=SimpleNamespace(
                get_scoped=AsyncMock(return_value=policy)
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        codec = MutationGrantCodec(
            secret="mutation-grant-test-secret-at-least-32-bytes",
            issuer="kbot-aiops-api",
            audience="kbot-aiops-db-executor",
        )
        service = AIOpsChangeService(
            uow_factory=lambda: context,
            action_registry=registry,
            approval_enabled=True,
            mutation_enabled=True,
            mutation_grant_codec=codec,
        )
        command = MutationClaimRequest(
            executor_request_id=request_id,
            executor_instance_id="executor-01",
            action_catalog_hash=registry.catalog_hash,
        )
        first = asyncio.run(
            service.claim_execution(
                execution_id=execution_id,
                command=command,
                trace_id="trace-claim",
            )
        )
        self.assertEqual(first.status, "SUBMITTED")
        self.assertEqual(token.status, "CONSUMED")
        self.assertEqual(proposal.status, "CONSUMED")
        self.assertEqual(execution.status, "SUBMITTED")
        self.assertEqual(execution.executor_instance_id, "executor-01")
        decoded = codec.verify(first.grant, now=now)
        self.assertEqual(decoded.execution_id, execution_id)
        self.assertEqual(decoded.max_database_attempts, 1)
        self.assertEqual(
            decoded.execution_credential_id,
            target.execution_credential_id,
        )
        second = asyncio.run(
            service.claim_execution(
                execution_id=execution_id,
                command=command,
                trace_id="trace-claim-retry",
            )
        )
        self.assertEqual(first.grant, second.grant)
        self.assertEqual(uow.commit.await_count, 1)


class ExecutionCallbackTest(unittest.TestCase):
    def test_callback_applies_monotonic_running_and_terminal_states(
        self,
    ) -> None:
        now = datetime.now(UTC)
        execution_id = uuid7()
        request_id = uuid7()
        run_id = uuid7()
        task_id = uuid7()
        proposal_id = uuid7()
        target_id = uuid7()
        artifact_id = uuid7()
        grant_hash = hashlib.sha256(
            json.dumps(str(execution_id)).encode()
        ).hexdigest()
        execution = SimpleNamespace(
            execution_id=execution_id,
            executor_request_id=str(request_id),
            executor_instance_id="executor-01",
            grant_jti_hash=grant_hash,
            ops_run_id=run_id,
            ops_task_id=task_id,
            proposal_id=proposal_id,
            approval_token_id=uuid7(),
            target_id=target_id,
            proposal_hash="a" * 64,
            command_hash="b" * 64,
            status="SUBMITTED",
            status_version=2,
            claimed_at=now,
            started_at=None,
            completed_at=None,
            result_artifact_id=None,
            result_hash=None,
            error_code=None,
            error_message=None,
            updated_at=now,
        )
        run = SimpleNamespace(
            ops_run_id=run_id,
            actor_id="portal:user-1",
            agent_id=uuid7(),
            target_id=target_id,
            plan_snapshot_json={
                "target": {
                                        "domain_id": 200,
                    "security_level": 3,
                }
            },
        )
        proposal = SimpleNamespace(
            proposal_id=proposal_id,
            ops_task_id=task_id,
        )
        inboxes = []

        async def add_inbox(entity):
            inboxes.append(entity)
            return entity

        async def add_artifact(entity):
            entity.artifact_id = artifact_id
            return entity

        uow = SimpleNamespace(
            inbox=SimpleNamespace(
                get_by_message=AsyncMock(return_value=None),
                add=AsyncMock(side_effect=add_inbox),
            ),
            changes=SimpleNamespace(
                get_execution=AsyncMock(return_value=execution),
                get_proposal=AsyncMock(return_value=proposal),
                get_approval_token=AsyncMock(
                    return_value=SimpleNamespace()
                ),
            ),
            runs=SimpleNamespace(
                database_now=AsyncMock(return_value=now),
                get_run=AsyncMock(return_value=run),
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            commit=AsyncMock(),
        )
        context = AsyncMock()
        context.__aenter__.return_value = uow
        service = AIOpsChangeService(
            uow_factory=lambda: context,
            action_registry=ActionRegistry.load(),
            approval_enabled=True,
            mutation_enabled=True,
        )
        running = ExecutionStatusEvent(
            event_id=uuid7(),
            executor_request_id=request_id,
            execution_id=execution_id,
            executor_instance_id="executor-01",
            grant_jti_hash=grant_hash,
            status_version=3,
            status="RUNNING",
            occurred_at=now,
        )
        first = asyncio.run(
            service.apply_execution_event(
                event=running, trace_id="trace-callback"
            )
        )
        self.assertTrue(first.accepted)
        self.assertEqual(execution.status, "RUNNING")
        result_body = {
            "accepted": True,
            "action_template_id": "db.session.terminate",
            "affected_object_count": 1,
        }
        result_hash = hashlib.sha256(
            json.dumps(
                result_body,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        terminal = ExecutionStatusEvent(
            event_id=uuid7(),
            executor_request_id=request_id,
            execution_id=execution_id,
            executor_instance_id="executor-01",
            grant_jti_hash=grant_hash,
            status_version=4,
            status="SUCCEEDED",
            occurred_at=now,
            bounded_result=result_body,
            result_hash=result_hash,
        )
        second = asyncio.run(
            service.apply_execution_event(
                event=terminal, trace_id="trace-callback"
            )
        )
        self.assertTrue(second.accepted)
        self.assertEqual(execution.status, "SUCCEEDED")
        self.assertEqual(execution.status_version, 4)
        self.assertEqual(execution.result_artifact_id, artifact_id)
        stored_artifact = uow.runs.add_artifact.await_args.args[0]
        self.assertEqual(
            stored_artifact.payload_json["proposal_id"],
            str(proposal_id),
        )
        self.assertEqual(len(inboxes), 2)
        self.assertTrue(all(item.status == "PROCESSED" for item in inboxes))
        uow.outbox.add.assert_awaited_once()
        outbox = uow.outbox.add.await_args.args[0]
        self.assertEqual(
            outbox.event_type, "OPS_EXECUTION_VERIFY_REQUESTED"
        )


if __name__ == "__main__":
    unittest.main()
