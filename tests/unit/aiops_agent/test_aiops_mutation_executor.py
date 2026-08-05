"""9C3 隔离 Executor 单次变更闸门测试。"""

from __future__ import annotations

import asyncio
import hashlib
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiops_agent.actions import (
    ActionRegistry,
    ActionRenderer,
    MutationGrantCodec,
)
from aiops_agent.executor import (
    MutationExecutionError,
    MutationExecutorService,
)
from aiops_agent.executor.drivers import (
    MutationDriverError,
    MutationDriverResult,
)
from platform_core.contracts.aiops.executor import (
    MutationClaimReceipt,
    MutationExecutionGrant,
    MutationExecutionRequest,
)
from platform_core.contracts.aiops.internal import EventReceipt
from platform_core.identity import uuid7


class _ControlPlane:
    def __init__(self, receipt, *, reject_running: bool = False):
        self.receipt = receipt
        self.reject_running = reject_running
        self.events = []

    async def claim_execution(self, execution_id, request, *, trace_id):
        del execution_id, request, trace_id
        return self.receipt

    async def publish_event(self, event, *, trace_id):
        del trace_id
        self.events.append(event)
        accepted = not (
            self.reject_running and event.status == "RUNNING"
        )
        return EventReceipt(
            event_id=event.event_id,
            accepted=accepted,
        )

    async def issue_credential(self, grant, *, trace_id):
        del grant, trace_id
        return SimpleNamespace(username="ops", password="secret")


class _Driver:
    db_type = "MYSQL"

    def __init__(self, error: MutationDriverError | None = None):
        self.error = error
        self.calls = 0

    async def execute_action(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        return MutationDriverResult(
            bounded_result={
                "accepted": True,
                "action_template_id": "db.session.terminate",
                "affected_object_count": 1,
            }
        )


class MutationExecutorServiceTest(unittest.TestCase):
    def _fixture(self, *, driver=None, reject_running=False):
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
        action = ActionRenderer().render(
            template, {"session_id": 42}
        )
        codec = MutationGrantCodec(
            secret="mutation-grant-test-secret-at-least-32-bytes",
            issuer="kbot-aiops-api",
            audience="kbot-aiops-db-executor",
        )
        execution_id = uuid7()
        request_id = uuid7()
        now = datetime.now(UTC)
        grant = MutationExecutionGrant(
            issuer="kbot-aiops-api",
            audience="kbot-aiops-db-executor",
            grant_id=execution_id,
            issued_at=now,
            expires_at=now + timedelta(seconds=30),
            execution_id=execution_id,
            executor_request_id=request_id,
            executor_instance_id="executor-test",
            target_id=uuid7(),
            domain_id=100,
            target_version=1,
            db_type="MYSQL",
            connection_profile={
                "host": "mysql.internal",
                "port": 3306,
                "database": "ops",
                "tls_enabled": True,
            },
            execution_credential_id=uuid7(),
            action_template_id=action.action_template_id,
            action_template_version=action.action_template_version,
            action_template_variant=action.variant,
            renderer_version=action.renderer_version,
            typed_parameters=action.typed_parameters,
            action_template_hash=action.template_hash,
            parameters_hash=action.parameters_hash,
            command_hash=action.command_hash,
            proposal_hash="a" * 64,
            policy_decision_hash="b" * 64,
            approval_token_hash="c" * 64,
            approver_id="portal:user-1",
            action_catalog_hash=registry.catalog_hash,
            statement_timeout_seconds=60,
            trace_id="trace-source",
        )
        receipt = MutationClaimReceipt(
            execution_id=execution_id,
            executor_request_id=request_id,
            status="SUBMITTED",
            grant=codec.issue(grant),
            expires_at=grant.expires_at,
        )
        control = _ControlPlane(
            receipt, reject_running=reject_running
        )
        resolved_driver = driver or _Driver()
        service = MutationExecutorService(
            enabled=True,
            executor_instance_id="executor-test",
            registry=registry,
            grant_codec=codec,
            control_plane=control,
            drivers=(resolved_driver,),
            concurrency=1,
        )
        request = MutationExecutionRequest(
            execution_id=execution_id,
            executor_request_id=request_id,
            idempotency_key=f"execution:{execution_id}:dispatch",
        )
        return service, request, control, resolved_driver

    def test_running_is_persisted_before_exactly_one_database_call(
        self,
    ) -> None:
        service, request, control, driver = self._fixture()
        result = asyncio.run(
            service.execute(request, trace_id="trace-dispatch")
        )
        self.assertEqual(result.status, "SUCCEEDED")
        self.assertEqual(driver.calls, 1)
        self.assertEqual(
            [event.status for event in control.events],
            ["RUNNING", "SUCCEEDED"],
        )
        self.assertEqual(
            [event.status_version for event in control.events],
            [3, 4],
        )

    def test_rejected_running_event_prevents_database_call(self) -> None:
        service, request, _, driver = self._fixture(
            reject_running=True
        )
        with self.assertRaises(MutationExecutionError) as caught:
            asyncio.run(
                service.execute(request, trace_id="trace-dispatch")
            )
        self.assertEqual(caught.exception.code, "RUNNING_EVENT_REJECTED")
        self.assertEqual(driver.calls, 0)

    def test_uncertain_driver_outcome_is_reported_as_unknown(self) -> None:
        driver = _Driver(
            MutationDriverError(
                "EXECUTION_OUTCOME_UNKNOWN",
                outcome_unknown=True,
            )
        )
        service, request, control, _ = self._fixture(driver=driver)
        result = asyncio.run(
            service.execute(request, trace_id="trace-dispatch")
        )
        self.assertEqual(result.status, "UNKNOWN")
        terminal = control.events[-1]
        self.assertEqual(terminal.status, "UNKNOWN")
        self.assertEqual(
            terminal.result_hash,
            hashlib.sha256(
                b'{"accepted":false,"action_template_id":'
                b'"db.session.terminate","outcome_unknown":true}'
            ).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
