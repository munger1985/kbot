"""AIOps Run 最终输出读取边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


class AIOpsRunResultTest(unittest.TestCase):
    def test_returns_scoped_final_artifact_payload(self) -> None:
        run_id = uuid7()
        artifact_id = uuid7()
        now = datetime.now(UTC)
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="COMPLETED",
            root_cause_level="PROBABLE",
            final_artifact_id=artifact_id,
            completed_at=now,
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id,
            ops_run_id=run_id,
            artifact_type="DIAGNOSIS_REPORT",
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            content_hash="a" * 64,
            payload_json={"status": "READY", "facts": []},
        )
        runs = SimpleNamespace(
            get_run_scoped=AsyncMock(return_value=run),
            get_artifact=AsyncMock(return_value=artifact),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(SimpleNamespace(runs=runs)),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )

        result = asyncio.run(
            service.get_run_result(
                ops_run_id=run_id,
                domain_id=200,
            )
        )

        self.assertEqual(result.payload["status"], "READY")
        self.assertEqual(result.final_artifact.artifact_id, artifact_id)
        runs.get_run_scoped.assert_awaited_once_with(
            ops_run_id=run_id,
            domain_id=200,
        )

    def test_rejects_artifact_not_owned_by_run(self) -> None:
        run_id = uuid7()
        run = SimpleNamespace(
            ops_run_id=run_id,
            status="COMPLETED",
            root_cause_level="INCONCLUSIVE",
            final_artifact_id=uuid7(),
            completed_at=datetime.now(UTC),
        )
        artifact = SimpleNamespace(
            artifact_id=run.final_artifact_id,
            ops_run_id=uuid7(),
            artifact_type="DIAGNOSIS_REPORT",
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            content_hash="b" * 64,
            payload_json={"status": "READY"},
        )
        runs = SimpleNamespace(
            get_run_scoped=AsyncMock(return_value=run),
            get_artifact=AsyncMock(return_value=artifact),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(SimpleNamespace(runs=runs)),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )

        result = asyncio.run(
            service.get_run_result(
                ops_run_id=run_id,
                domain_id=200,
            )
        )

        self.assertIsNone(result.final_artifact)
        self.assertIsNone(result.payload)


if __name__ == "__main__":
    unittest.main()
