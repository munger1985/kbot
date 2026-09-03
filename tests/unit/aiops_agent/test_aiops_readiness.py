"""AIOps Schema 就绪门禁与不可重试错误回归测试。"""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.exc import IntegrityError

from aiops_agent.application.errors import AIOpsSchemaNotReadyError
from aiops_agent.application.investigation.errors import TurnPlanningStageError
from aiops_agent.application.investigation.service import TurnPlanningService
from aiops_agent.bootstrap.common import AIOpsProcessRuntime
from aiops_agent.workers.outbox_dispatcher import AIOpsOutboxDispatcher
from platform_core.identity import uuid7


class _ScalarResult:
    def __init__(self, value) -> None:
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _SchemaSession:
    def __init__(self, values) -> None:
        self._values = list(values)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def execute(self, _statement):
        return _ScalarResult(self._values.pop(0))


class _OutboxRepository:
    def __init__(self) -> None:
        self.message = SimpleNamespace(
            outbox_id=uuid7(),
            event_type="aiops.turn.understanding_requested",
            payload_json={"domain_id": 7, "turn_id": str(uuid7())},
            attempt_count=1,
            max_attempts=3,
        )
        self.release = None

    async def recover_expired(self, **_):
        return False

    async def claim(self, **_):
        message, self.message = self.message, None
        return message

    async def release_failed(self, **kwargs):
        self.release = dict(kwargs)
        return True


class _OutboxUow:
    def __init__(self) -> None:
        self.outbox = _OutboxRepository()
        self.runs = SimpleNamespace(database_now=self._database_now)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def _database_now(self):
        return datetime.now(UTC)

    async def commit(self):
        return None


class _IntegrityFailureSink:
    def __init__(self) -> None:
        self.terminal_failures = []

    async def publish(self, _event_type, _payload):
        raise IntegrityError(
            "INSERT INTO KBOT_OPS_TASK",
            {},
            RuntimeError("ORA-02290: check constraint violated"),
        )

    async def on_terminal_failure(self, event_type, payload, exc):
        self.terminal_failures.append(
            (event_type, dict(payload), type(exc).__name__)
        )


class AIOpsReadinessTest(unittest.IsolatedAsyncioTestCase):
    async def test_ready_requires_schema_21_contract_integrity(self) -> None:
        session = _SchemaSession((1, 8, 1, 1, 1, 1))
        runtime = AIOpsProcessRuntime(
            settings=object(),
            service_name="test-aiops",
            database_runtime=SimpleNamespace(
                session_factory=lambda: session
            ),
        )

        checks = await runtime.check_aiops_schema()

        self.assertEqual(
            {"aiops_schema": "ok", "aiops_schema_integrity": "ok"},
            checks,
        )

    async def test_ready_rejects_partial_schema_21_contract(self) -> None:
        session = _SchemaSession((1, 7, 1, 1, 1, 1))
        runtime = AIOpsProcessRuntime(
            settings=object(),
            service_name="test-aiops",
            database_runtime=SimpleNamespace(
                session_factory=lambda: session
            ),
        )

        checks = await runtime.check_aiops_schema()

        self.assertEqual("ok", checks["aiops_schema"])
        self.assertEqual(
            "contract_mismatch", checks["aiops_schema_integrity"]
        )

    async def test_ready_rejects_dynamic_tool_class_constraint_drift(
        self,
    ) -> None:
        session = _SchemaSession((1, 8, 1, 1, 1, 0))
        runtime = AIOpsProcessRuntime(
            settings=object(),
            service_name="test-aiops",
            database_runtime=SimpleNamespace(
                session_factory=lambda: session
            ),
        )

        checks = await runtime.check_aiops_schema()

        self.assertEqual("ok", checks["aiops_schema"])
        self.assertEqual(
            "contract_mismatch", checks["aiops_schema_integrity"]
        )

    async def test_ready_rejects_non_clob_report_summary(self) -> None:
        session = _SchemaSession((1, 8, 1, 0, 1, 1))
        runtime = AIOpsProcessRuntime(
            settings=object(),
            service_name="test-aiops",
            database_runtime=SimpleNamespace(
                session_factory=lambda: session
            ),
        )

        checks = await runtime.check_aiops_schema()

        self.assertEqual("ok", checks["aiops_schema"])
        self.assertEqual(
            "contract_mismatch", checks["aiops_schema_integrity"]
        )

    async def test_schema_gate_stops_before_planning_model(self) -> None:
        service = object.__new__(TurnPlanningService)
        service._schema_ready_check = AsyncMock(
            return_value={
                "aiops_schema": "ok",
                "aiops_schema_integrity": "contract_mismatch",
            }
        )
        service._execute_once = AsyncMock()

        with self.assertRaises(AIOpsSchemaNotReadyError):
            await service.execute({"domain_id": 7, "turn_id": str(uuid7())})

        service._execute_once.assert_not_awaited()

    async def test_integrity_error_is_terminal_on_first_attempt(self) -> None:
        uow = _OutboxUow()
        sink = _IntegrityFailureSink()
        dispatcher = AIOpsOutboxDispatcher(
            uow_factory=lambda: uow,
            sink=sink,
            dispatcher_id="test-outbox",
            lease_seconds=30,
            interval_seconds=1,
        )

        worked = await dispatcher.run_once()

        self.assertTrue(worked)
        self.assertEqual("FAILED", uow.outbox.release["new_status"])
        self.assertEqual(
            "AIOPS_SCHEMA_INTEGRITY_ERROR",
            uow.outbox.release["error_code"],
        )
        self.assertEqual(1, len(sink.terminal_failures))

    def test_planning_integrity_error_keeps_safe_terminal_semantics(self) -> None:
        error = IntegrityError(
            "INSERT INTO KBOT_OPS_TASK",
            {},
            RuntimeError("ORA-02290: check constraint violated"),
        )

        wrapped = TurnPlanningStageError(error)

        self.assertFalse(wrapped.retryable)
        self.assertEqual("AIOPS_SCHEMA_INTEGRITY_ERROR", wrapped.code)
        self.assertEqual("database-contract-violation", wrapped.safe_detail)


if __name__ == "__main__":
    unittest.main()
