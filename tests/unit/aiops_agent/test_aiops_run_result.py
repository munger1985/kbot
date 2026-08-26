"""AIOps Run 最终输出读取边界测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.runtime.service import (
    AIOpsRuntimeService,
    _diagnosis_answer_markdown,
)
from platform_core.identity import uuid7


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


class AIOpsRunResultTest(unittest.TestCase):
    def test_chat_diagnosis_has_natural_streaming_markdown_projection(self) -> None:
        content = _diagnosis_answer_markdown(
            {
                "root_cause": {"effective_level": "PROBABLE"},
                "diagnosis_rationale": "锁等待导致响应变慢",
                "facts": [{"fact_summary": "阻塞会话持续存在"}],
                "solution": {
                    "immediate_mitigations": ["确认阻塞源"],
                    "long_term_remediations": ["缩短事务边界"],
                },
                "gaps": ["缺少应用调用链"],
            }
        )

        self.assertIn("锁等待导致响应变慢", content)
        self.assertNotIn("根因等级", content)
        self.assertNotIn("已验证事实", content)
        self.assertNotIn("## 诊断结论", content)
        self.assertIn("接下来可以这样处理", content)
        self.assertIn("- 确认阻塞源", content)

    def test_direct_answer_does_not_append_report_sections(self) -> None:
        content = _diagnosis_answer_markdown(
            {
                "direct_answer": {
                    "answer_text": "| 表空间 | 使用率 |\n| --- | --- |\n| USERS | 80% |",
                    "limitations": [],
                },
                "facts": [{"fact_summary": "USERS 使用率为 80%"}],
                "solution": {"immediate_mitigations": ["无需处理"]},
            }
        )

        self.assertIn("| USERS | 80% |", content)
        self.assertNotIn("无需处理", content)
        self.assertNotIn("已验证事实", content)

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
