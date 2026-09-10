"""Oracle 原生工作负载报告下载边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
from platform_core.contracts.aiops.executor import (
    DatabaseColumn,
    DatabaseObservation,
)
from platform_core.contracts.aiops.types import MeasurementSemantics
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


class WorkloadReportDownloadTest(unittest.TestCase):
    def test_download_reassembles_only_complete_authorized_report(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        observation = DatabaseObservation(
            executor_request_id=uuid7(), target_id=uuid7(),
            tool_id="db.oracle.awr.report", tool_version="1.0.0",
            variant="oracle_19_plus_awr_report_html", template_sha256="a" * 64,
            db_type="ORACLE", db_version="19.0.0.0.0",
            capability_snapshot_hash="b" * 64, captured_at=datetime.now(UTC),
            duration_ms=10,
            columns=(DatabaseColumn(name="output", logical_type="STRING", sensitivity="PUBLIC"),),
            rows=(("<html>",), ("报告</html>",)), row_count=2,
            truncated=False, result_sha256="c" * 64, parameters_sha256="d" * 64,
        )
        payload = DbaToolResult(
            source_type="TOOL", source_id="db.oracle.awr.report",
            source_version="1.0.0", definition_hash="e" * 64,
            output_schema="DBA_TOOL_RESULT.v1",
            measurement_semantics=MeasurementSemantics.HISTORICAL_SAMPLES,
            presentation_kind="TABLE", status="SUCCEEDED",
            tool_outcomes=(ToolOutcome(
                step_id="report", tool_id="db.oracle.awr.report",
                tool_version="1.0.0", status="SUCCEEDED",
                observation=observation,
            ),),
        ).model_dump(mode="json")
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id=conversation_id)),
                get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id)),
            ),
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=[
                SimpleNamespace(artifact_id=artifact_id, schema_version="DBA_TOOL_RESULT.v1", payload_json=payload)
            ])),
        )
        service = ConversationTurnService(uow_factory=lambda: _context(uow))

        content = asyncio.run(service.get_workload_report_content(
            domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
            actor_id="user-1", tool_id="db.oracle.awr.report",
        ))

        self.assertEqual(b"<html>\xe6\x8a\xa5\xe5\x91\x8a</html>", content)
