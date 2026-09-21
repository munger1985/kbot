"""Oracle 原生工作负载报告下载边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.errors import AIOpsApplicationError
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


def _observation(tool_id: str, body: str) -> DatabaseObservation:
    return DatabaseObservation(
        executor_request_id=uuid7(), target_id=uuid7(),
        tool_id=tool_id, tool_version="1.0.0",
        variant="oracle_19_plus_html", template_sha256="a" * 64,
        db_type="ORACLE", db_version="19.0.0.0.0",
        capability_snapshot_hash="b" * 64, captured_at=datetime.now(UTC),
        duration_ms=10,
        columns=(DatabaseColumn(name="output", logical_type="STRING", sensitivity="PUBLIC"),),
        rows=((body,),), row_count=1,
        truncated=False, result_sha256="c" * 64, parameters_sha256="d" * 64,
    )


def _payload(tool_id: str, action_id: str, body: str) -> dict:
    return DbaToolResult(
        source_type="TOOL", source_id=tool_id,
        source_version="1.0.0", definition_hash="e" * 64,
        output_schema="DBA_TOOL_RESULT.v1",
        measurement_semantics=MeasurementSemantics.HISTORICAL_SAMPLES,
        presentation_kind="TABLE", status="SUCCEEDED",
        tool_outcomes=(ToolOutcome(
            step_id=action_id, tool_id=tool_id,
            tool_version="1.0.0", status="SUCCEEDED",
            observation=_observation(tool_id, body),
        ),),
    ).model_dump(mode="json")


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
            rows=(("<html>",), (None,), ("报告</html>",)), row_count=3,
            truncated=False, result_sha256="c" * 64, parameters_sha256="d" * 64,
        )
        payload = DbaToolResult(
            source_type="TOOL", source_id="db.oracle.awr.report",
            source_version="1.0.0", definition_hash="e" * 64,
            output_schema="DBA_TOOL_RESULT.v1",
            measurement_semantics=MeasurementSemantics.HISTORICAL_SAMPLES,
            presentation_kind="TABLE", status="SUCCEEDED",
            tool_outcomes=(ToolOutcome(
                step_id="a1", tool_id="db.oracle.awr.report",
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
            action_id="a1",
        ))

        self.assertEqual(b"<html>\xe6\x8a\xa5\xe5\x91\x8a</html>", content)

    def test_download_selects_each_same_tool_report_by_action(self) -> None:
        conversation_id, turn_id, run_id = uuid7(), uuid7(), uuid7()
        artifacts = [
            SimpleNamespace(
                artifact_id=uuid7(),
                schema_version="DBA_TOOL_RESULT.v1",
                payload_json=_payload(
                    "db.oracle.awr.report", "a2", "<html>yesterday</html>",
                ),
            ),
            SimpleNamespace(
                artifact_id=uuid7(),
                schema_version="DBA_TOOL_RESULT.v1",
                payload_json=_payload(
                    "db.oracle.awr.report", "a3", "<html>today</html>",
                ),
            ),
            SimpleNamespace(
                artifact_id=uuid7(),
                schema_version="DBA_TOOL_RESULT.v1",
                payload_json=_payload(
                    "db.oracle.awr.diff_report", "a4", "<html>diff</html>",
                ),
            ),
        ]
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id=conversation_id)),
                get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id)),
            ),
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=artifacts)),
        )
        service = ConversationTurnService(uow_factory=lambda: _context(uow))

        yesterday = asyncio.run(service.get_workload_report_content(
            domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
            actor_id="user-1", tool_id="db.oracle.awr.report",
            action_id="a2",
        ))
        today = asyncio.run(service.get_workload_report_content(
            domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
            actor_id="user-1", tool_id="db.oracle.awr.report",
            action_id="a3",
        ))
        diff = asyncio.run(service.get_workload_report_content(
            domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
            actor_id="user-1", tool_id="db.oracle.awr.diff_report",
            action_id="a4",
        ))

        self.assertEqual(b"<html>yesterday</html>", yesterday)
        self.assertEqual(b"<html>today</html>", today)
        self.assertEqual(b"<html>diff</html>", diff)
        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(service.get_workload_report_content(
                domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
                actor_id="user-1", tool_id="db.oracle.awr.report",
                action_id="a4",
            ))


    def test_download_allows_sql_monitor_report(self) -> None:
        conversation_id, turn_id, run_id = uuid7(), uuid7(), uuid7()
        artifacts = [
            SimpleNamespace(
                artifact_id=uuid7(),
                schema_version="DBA_TOOL_RESULT.v1",
                payload_json=_payload(
                    "db.oracle.sql_monitor.report",
                    "a5",
                    "<html>sql-monitor</html>",
                ),
            ),
        ]
        uow = SimpleNamespace(
            conversations=SimpleNamespace(get_conversation=AsyncMock(
                return_value=SimpleNamespace(created_by="user-1")
            )),
            turns=SimpleNamespace(
                get_turn=AsyncMock(return_value=SimpleNamespace(conversation_id=conversation_id)),
                get_run_link=AsyncMock(return_value=SimpleNamespace(ops_run_id=run_id)),
            ),
            runs=SimpleNamespace(list_artifacts=AsyncMock(return_value=artifacts)),
        )
        service = ConversationTurnService(uow_factory=lambda: _context(uow))

        content = asyncio.run(service.get_workload_report_content(
            domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
            actor_id="user-1", tool_id="db.oracle.sql_monitor.report",
            action_id="a5",
        ))
        self.assertEqual(b"<html>sql-monitor</html>", content)
        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(service.get_workload_report_content(
                domain_id=1, conversation_id=conversation_id, turn_id=turn_id,
                actor_id="user-1", tool_id="db.sql.plan_monitor",
                action_id="a5",
            ))
