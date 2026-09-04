"""正式报告模板、展示投影与 PDF 渲染的无数据库测试。"""

from __future__ import annotations

import unittest
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.reporting import (
    SYSTEM_REPORT_TEMPLATES,
    closed_period_window,
    render_pdf,
    report_presentation,
    normalize_report_source,
    resolve_historical_report_template,
    resolve_report_template_reference,
    validate_template_definition,
)
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.identity import uuid7


class FormalReportingTest(unittest.TestCase):
    def test_system_templates_cover_requested_periods(self) -> None:
        self.assertEqual(
            {
                "system:diagnosis.standard",
                "system:inspection.daily",
                "system:inspection.monthly",
                "system:inspection.quarterly",
                "system:inspection.annual",
            },
            set(SYSTEM_REPORT_TEMPLATES),
        )

    def test_historical_system_template_reference_is_resolvable(self) -> None:
        self.assertEqual(
            SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"],
            resolve_report_template_reference("diagnosis.standard"),
        )

    def test_historical_daily_inspection_template_is_resolvable(self) -> None:
        self.assertEqual(
            SYSTEM_REPORT_TEMPLATES["system:inspection.daily"],
            resolve_historical_report_template(
                template_ref="01a00000-0000-7000-8000-000000000000",
                report_type="INSPECTION_DAILY",
            ),
        )

    def test_custom_template_cannot_hide_evidence_boundary(self) -> None:
        with self.assertRaises(AIOpsApplicationError):
            validate_template_definition({
                "sections": [{"kind": "EXECUTIVE_SUMMARY"}],
            })

    def test_presentation_and_pdf_share_the_frozen_template(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"]
        presentation = report_presentation(
            template=template,
            payload={
                "title": "数据库故障诊断报告",
                "status": "PARTIAL",
                "summary": "已确认锁等待持续升高。",
                "period_start": "2026-09-01T00:00:00+00:00",
                "period_end": "2026-09-01T01:00:00+00:00",
                "scope": {"root_cause_grade": "PROBABLE"},
                "facts": [{"summary": "锁等待持续升高"}],
                "gaps": [{"code": "MISSING_ASH"}],
                "recommendations": ["确认阻塞会话"],
                "evidence_refs": [{"artifact_id": "evidence-1", "content_hash": "a" * 64}],
            },
        )
        kinds = [item["kind"] for item in presentation["sections"]]
        self.assertIn("EVIDENCE_BOUNDARY", kinds)
        pdf = render_pdf(presentation)
        self.assertTrue(pdf.startswith(b"%PDF-"))
        # 标准生成器必须嵌入字体和 Unicode 映射，不能依赖阅读器安装中文字体。
        self.assertIn(b"/FontFile2", pdf)
        self.assertNotIn(b"FEFF", pdf)
        self.assertIn(b"/ToUnicode", pdf)

    def test_inspection_result_uses_the_same_source_normalizer(self) -> None:
        source = normalize_report_source(
            schema_version="DB_DIAGNOSTIC_REPORT.v1",
            source_kind="INSPECTION",
            payload={
                "status": "PARTIAL",
                "observation_count": 8,
                "gap_count": 1,
                "tools": ["db.instance.identity"],
                "gaps": [{"code": "CAPABILITY_UNAVAILABLE"}],
            },
        )
        self.assertEqual("PARTIAL", source["status"])
        self.assertEqual("INCONCLUSIVE", source["root_cause"]["effective_level"])
        self.assertEqual("已完成 8 项观测", source["facts"][0]["summary"])

    def test_closed_monthly_window_uses_report_timezone(self) -> None:
        start, end = closed_period_window(
            period_kind="MONTHLY",
            timezone="Asia/Shanghai",
            now=datetime(2026, 3, 1, 1, tzinfo=UTC),
        )
        self.assertEqual(datetime(2026, 1, 31, 16, tzinfo=UTC), start)
        self.assertEqual(datetime(2026, 2, 28, 16, tzinfo=UTC), end)

    def test_periodic_inspection_aggregates_all_frozen_sources(self) -> None:
        first_id, second_id = uuid7(), uuid7()
        first_artifact, second_artifact = uuid7(), uuid7()
        runs = [
            SimpleNamespace(ops_run_id=first_id, final_artifact_id=first_artifact, completed_at=datetime(2026, 2, 2, tzinfo=UTC)),
            SimpleNamespace(ops_run_id=second_id, final_artifact_id=second_artifact, completed_at=datetime(2026, 2, 3, tzinfo=UTC)),
        ]
        artifacts = {
            first_artifact: SimpleNamespace(artifact_id=first_artifact, content_hash="a" * 64, schema_version="DB_DIAGNOSTIC_REPORT.v1", payload_json={"status": "READY", "observation_count": 4, "gap_count": 0, "tools": []}),
            second_artifact: SimpleNamespace(artifact_id=second_artifact, content_hash="b" * 64, schema_version="DB_DIAGNOSTIC_REPORT.v1", payload_json={"status": "PARTIAL", "observation_count": 3, "gap_count": 1, "tools": [], "gaps": [{"code": "MISSING_ASH"}]}),
        }
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts[artifact_id])))
        service = AIOpsRuntimeService(uow_factory=AsyncMock(), blueprint_registry=AsyncMock(), handler_registry=AsyncMock())
        result = asyncio.run(service._aggregate_inspection_sources(uow=uow, runs=runs, period_kind="MONTHLY", period_start=datetime(2026, 1, 31, 16, tzinfo=UTC), period_end=datetime(2026, 2, 28, 16, tzinfo=UTC)))
        self.assertEqual("PARTIAL", result["status"])
        self.assertEqual(2, len(result["evidence_refs"]))
        self.assertIn("完成 2 次巡检", result["inspection_coverage"])


if __name__ == "__main__":
    unittest.main()
