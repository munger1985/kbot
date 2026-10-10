"""会话范围内统一 Artifact 下载边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.turns import ConversationTurnService
from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
from platform_core.contracts.aiops.executor import DatabaseColumn, DatabaseObservation
from platform_core.contracts.aiops.types import MeasurementSemantics
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


def _payload(tool_id: str, body: str, *, truncated: bool = False) -> dict:
    observation = DatabaseObservation(
        executor_request_id=uuid7(), target_id=uuid7(),
        tool_id=tool_id, tool_version="1.0.0",
        variant="oracle_19_plus_html", template_sha256="a" * 64,
        db_type="ORACLE", db_version="19.0.0.0.0",
        capability_snapshot_hash="b" * 64, captured_at=datetime.now(UTC),
        duration_ms=10,
        columns=(DatabaseColumn(
            name="output", logical_type="STRING", sensitivity="PUBLIC",
        ),),
        rows=((body,),), row_count=1, truncated=truncated,
        result_sha256="c" * 64, parameters_sha256="d" * 64,
    )
    return DbaToolResult(
        source_type="TOOL", source_id=tool_id,
        source_version="1.0.0", definition_hash="e" * 64,
        output_schema="DBA_TOOL_RESULT.v1",
        measurement_semantics=MeasurementSemantics.HISTORICAL_SAMPLES,
        presentation_kind="TABLE", status="SUCCEEDED",
        tool_outcomes=(ToolOutcome(
            step_id="a1", tool_id=tool_id, tool_version="1.0.0",
            status="SUCCEEDED", observation=observation,
        ),),
    ).model_dump(mode="json")


def _uow(*, conversation_id, actor_id, run_id, artifact):
    return SimpleNamespace(
        conversations=SimpleNamespace(get_conversation=AsyncMock(
            return_value=SimpleNamespace(created_by=actor_id)
        )),
        turns=SimpleNamespace(
            get_turn=AsyncMock(return_value=SimpleNamespace(
                conversation_id=conversation_id
            )),
            get_run_link=AsyncMock(return_value=SimpleNamespace(
                ops_run_id=run_id
            )),
        ),
        runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)),
    )


class ArtifactDownloadTest(unittest.TestCase):
    def test_download_reassembles_authorized_html_artifact(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id, ops_run_id=run_id,
            artifact_type="DBA_TOOL_RESULT",
            schema_version="DBA_TOOL_RESULT.v1",
            payload_json=_payload("db.oracle.awr.report", "<html>报告</html>"),
            payload_uri=None, provenance_json={},
            content_hash="a" * 64, byte_size=10,
        )
        service = ConversationTurnService(
            uow_factory=lambda: _context(_uow(
                conversation_id=conversation_id, actor_id="user-1",
                run_id=run_id, artifact=artifact,
            ))
        )

        content, media_type, file_name = asyncio.run(
            service.get_artifact_content(
                domain_id=1, conversation_id=conversation_id,
                turn_id=turn_id, actor_id="user-1", artifact_id=artifact_id,
            )
        )

        self.assertEqual(b"<html>\xe6\x8a\xa5\xe5\x91\x8a</html>", content)
        self.assertEqual("text/html", media_type)
        self.assertTrue(file_name.endswith(".html"))

    def test_download_returns_generated_report_as_json(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id, ops_run_id=run_id,
            artifact_type="MYSQL_WORKLOAD_REPORT",
            schema_version="MYSQL_WORKLOAD_REPORT.v1",
            payload_json={"report_origin": "AIOPS_GENERATED", "status": "READY"},
            payload_uri=None,
            provenance_json={"file_name": "mysql-workload.json"},
            content_hash="a" * 64, byte_size=10,
        )
        service = ConversationTurnService(
            uow_factory=lambda: _context(_uow(
                conversation_id=conversation_id, actor_id="user-1",
                run_id=run_id, artifact=artifact,
            ))
        )

        content, media_type, file_name = asyncio.run(
            service.get_artifact_content(
                domain_id=1, conversation_id=conversation_id,
                turn_id=turn_id, actor_id="user-1", artifact_id=artifact_id,
            )
        )

        self.assertEqual("application/json", media_type)
        self.assertEqual("mysql-workload.json", file_name)
        self.assertIn(b"AIOPS_GENERATED", content)

    def test_download_reads_external_payload_with_integrity_metadata(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id, ops_run_id=run_id,
            artifact_type="REPORT_PDF", schema_version="REPORT_PDF.v1",
            payload_json={}, payload_uri="artifact://report.pdf",
            provenance_json={
                "file_name": "database-report.pdf",
                "content_type": "application/pdf",
            },
            content_hash="a" * 64, byte_size=7,
        )
        upload_store = SimpleNamespace(read_artifact=Mock(return_value=b"PDFDATA"))
        service = ConversationTurnService(
            uow_factory=lambda: _context(_uow(
                conversation_id=conversation_id, actor_id="user-1",
                run_id=run_id, artifact=artifact,
            )),
            upload_store=upload_store,
        )

        content, media_type, file_name = asyncio.run(
            service.get_artifact_content(
                domain_id=1, conversation_id=conversation_id,
                turn_id=turn_id, actor_id="user-1", artifact_id=artifact_id,
            )
        )

        self.assertEqual(b"PDFDATA", content)
        self.assertEqual("application/pdf", media_type)
        self.assertEqual("database-report.pdf", file_name)
        upload_store.read_artifact.assert_called_once_with(
            payload_uri="artifact://report.pdf",
            content_hash="a" * 64,
            byte_size=7,
        )

    def test_download_rejects_cross_run_artifact(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id, ops_run_id=uuid7(),
            artifact_type="MYSQL_WORKLOAD_REPORT",
            schema_version="MYSQL_WORKLOAD_REPORT.v1", payload_json={},
            payload_uri=None, provenance_json={},
            content_hash="a" * 64, byte_size=2,
        )
        service = ConversationTurnService(
            uow_factory=lambda: _context(_uow(
                conversation_id=conversation_id, actor_id="user-1",
                run_id=run_id, artifact=artifact,
            ))
        )

        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(service.get_artifact_content(
                domain_id=1, conversation_id=conversation_id,
                turn_id=turn_id, actor_id="user-1", artifact_id=artifact_id,
            ))

    def test_download_rejects_truncated_html_artifact(self) -> None:
        conversation_id, turn_id, run_id, artifact_id = (
            uuid7(), uuid7(), uuid7(), uuid7()
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id, ops_run_id=run_id,
            artifact_type="DBA_TOOL_RESULT",
            schema_version="DBA_TOOL_RESULT.v1",
            payload_json=_payload(
                "db.oracle.ash.report", "<html>partial</html>", truncated=True,
            ),
            payload_uri=None, provenance_json={},
            content_hash="a" * 64, byte_size=10,
        )
        service = ConversationTurnService(
            uow_factory=lambda: _context(_uow(
                conversation_id=conversation_id, actor_id="user-1",
                run_id=run_id, artifact=artifact,
            ))
        )

        with self.assertRaises(AIOpsApplicationError):
            asyncio.run(service.get_artifact_content(
                domain_id=1, conversation_id=conversation_id,
                turn_id=turn_id, actor_id="user-1", artifact_id=artifact_id,
            ))


if __name__ == "__main__":
    unittest.main()
