"""AIOps 领导简报投影测试；不断言 SID / SQL 细节。"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.reporting import (
    SYSTEM_REPORT_TEMPLATES,
    render_pdf,
    report_presentation,
)
from aiops_agent.application.leadership import project_leadership_briefing
from aiops_agent.application.runtime.service import (
    AIOpsRuntimeService,
    sha256_json,
)
from platform_core.contracts.aiops.findings import (
    FindingCard,
    FindingCompilation,
    FindingConfirmation,
    FindingObjectRef,
    FindingSeverity,
    FindingType,
)
from platform_core.identity import uuid7


def _finding(
    *,
    finding_type: FindingType,
    severity: FindingSeverity,
    fields: dict | None = None,
    impact: str = "已确认影响范围",
    sql_id: str | None = "8ab12cd34ef56",
    instance_id: int | None = 1,
) -> FindingCard:
    return FindingCard(
        finding_id=f"{finding_type}:{severity}",
        finding_type=finding_type,
        severity=severity,
        confirmation=FindingConfirmation.CONFIRMED,
        object_ref=FindingObjectRef(
            object_kind="SESSION",
            sql_id=sql_id,
            instance_id=instance_id,
        ),
        fields=fields or {},
        impact=impact,
    )


def _dump(briefing: dict) -> str:
    return json.dumps(briefing, ensure_ascii=False, sort_keys=True)


class AIOpsLeadershipBriefingTest(unittest.TestCase):
    def test_lock_findings_omit_sid_and_sql_from_dump(self) -> None:
        briefing = project_leadership_briefing(
            payload={
                "report_type": "INCIDENT",
                "status": "READY",
                "summary": "根因等级：CONFIRMED。SID 88 被 SQL_ID 8ab12cd34ef56 阻塞。",
                "recommendations": ["确认阻塞会话后按审批执行"],
            },
            findings=(
                _finding(
                    finding_type=FindingType.LOCK_WAIT,
                    severity=FindingSeverity.HIGH,
                    fields={
                        "waiting_session_id": 123,
                        "blocking_session_id": 456,
                        "wait_seconds": 80,
                    },
                    impact="会话 123 已等待 80 秒，被会话 456 阻塞。",
                ),
            ),
        )
        dump = _dump(briefing)
        self.assertEqual(
            ["schema_version", "risk_level", "business_impact", "risks", "recommendations"],
            list(briefing),
        )
        self.assertEqual("HIGH", briefing["risk_level"])
        self.assertIn("业务事务可能被拉长等待", briefing["business_impact"][0])
        self.assertIn("超时", briefing["risks"][0])
        self.assertEqual(["确认阻塞会话后按审批执行"], briefing["recommendations"])
        self.assertNotIn("sql_id", dump)
        self.assertNotIn("instance_id", dump)
        self.assertNotIn("8ab12cd34ef56", dump)
        self.assertNotIn("SID 88", dump)
        self.assertNotIn("123", dump)
        self.assertNotIn("456", dump)

    def test_inspection_without_findings_stays_low_risk(self) -> None:
        briefing = project_leadership_briefing(
            payload={
                "report_type": "INSPECTION_DAILY",
                "status": "READY",
                "summary": "已完成 8 项观测，发现 0 个数据缺口",
                "scope": {"root_cause_grade": "INCONCLUSIVE"},
                "facts": [{"summary": "实例性能指标：检查已完成"}],
                "recommendations": ["继续按既定周期执行巡检。"],
            },
        )
        self.assertEqual("LOW", briefing["risk_level"])
        self.assertTrue(briefing["business_impact"])
        self.assertTrue(briefing["risks"])
        self.assertEqual(["继续按既定周期执行巡检。"], briefing["recommendations"])
        self.assertNotIn("sql_id", _dump(briefing))

    def test_fallback_summary_strips_session_identifiers(self) -> None:
        briefing = project_leadership_briefing(
            payload={
                "report_type": "INCIDENT",
                "status": "PARTIAL",
                "summary": "根因等级：INCONCLUSIVE。SID 88 与 sql_id=8ab12cd34ef56 仍待确认。",
                "scope": {"root_cause_grade": "INCONCLUSIVE"},
                "facts": [{"summary": "会话 88 仍在等待，instance_id=1"}],
                "recommendations": ["补充 ASH 后再判断 SID 88"],
            },
        )
        dump = _dump(briefing)
        self.assertEqual("MEDIUM", briefing["risk_level"])
        self.assertTrue(all("SID 88" not in item for item in briefing["business_impact"]))
        self.assertNotIn("8ab12cd34ef56", dump)
        self.assertNotIn("instance_id", dump)
        self.assertIn("证据不完整", "".join(briefing["risks"]))

    def test_tablespace_finding_exposes_capacity_not_sql(self) -> None:
        briefing = project_leadership_briefing(
            payload={
                "report_type": "INSPECTION_WEEKLY",
                "status": "READY",
                "recommendations": ["评估扩容窗口"],
            },
            findings=(
                _finding(
                    finding_type=FindingType.TABLESPACE,
                    severity=FindingSeverity.CRITICAL,
                    fields={"used_percent": 96.8, "tablespace_name": "USERS"},
                    sql_id=None,
                    instance_id=None,
                    impact="表空间 USERS 使用率 96.8%。",
                ),
                _finding(
                    finding_type=FindingType.WAIT_CLASS,
                    severity=FindingSeverity.MEDIUM,
                ),
            ),
        )
        self.assertEqual("CRITICAL", briefing["risk_level"])
        self.assertTrue(any("96.8" in item or "96" in item for item in briefing["business_impact"]))
        self.assertNotIn("sql_id", _dump(briefing))
        self.assertEqual(["评估扩容窗口"], briefing["recommendations"])

    def test_presentation_always_includes_briefing_without_replacing_summary(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"]
        presentation = report_presentation(
            template=template,
            payload={
                "title": "数据库故障诊断报告",
                "status": "READY",
                "summary": "根因等级：PROBABLE。锁等待持续升高。",
                "scope": {"root_cause_grade": "PROBABLE"},
                "facts": [{"summary": "锁等待持续升高"}],
                "recommendations": ["确认阻塞会话"],
            },
            findings=(
                _finding(
                    finding_type=FindingType.LOCK_WAIT,
                    severity=FindingSeverity.HIGH,
                ),
            ),
        )
        summary = next(
            item for item in presentation["sections"] if item["kind"] == "EXECUTIVE_SUMMARY"
        )
        self.assertEqual(["根因等级：PROBABLE。锁等待持续升高。"], summary["items"])
        briefing = presentation["leadership_briefing"]
        self.assertEqual("LEADERSHIP_BRIEFING.v1", briefing["schema_version"])
        self.assertEqual("HIGH", briefing["risk_level"])
        pdf = render_pdf(presentation)
        self.assertTrue(pdf.startswith(b"%PDF-"))

    def test_get_report_presentation_projects_findings_from_run(self) -> None:
        report_id, artifact_id, ops_run_id = uuid7(), uuid7(), uuid7()
        payload = {
            "title": "数据库故障诊断报告",
            "status": "READY",
            "summary": "根因等级：CONFIRMED。锁等待持续升高。",
            "ops_run_id": str(ops_run_id),
            "scope": {"root_cause_grade": "CONFIRMED"},
            "facts": [{"summary": "锁等待持续升高"}],
            "gaps": [],
            "recommendations": ["确认阻塞会话"],
            "provenance": {},
        }
        content_hash = sha256_json(payload)
        compilation = FindingCompilation(
            findings=(
                _finding(
                    finding_type=FindingType.LOCK_WAIT,
                    severity=FindingSeverity.HIGH,
                    impact="会话 99 已等待 12 秒，被会话 100 阻塞。",
                ),
            )
        )
        report = SimpleNamespace(
            report_id=report_id,
            content_artifact_id=artifact_id,
            template_id="01a00000-0000-7000-8000-000000000000",
            template_version="1.0.0",
            report_type="INCIDENT",
            ops_run_id=ops_run_id,
        )
        artifact = SimpleNamespace(
            artifact_id=artifact_id,
            content_hash=content_hash,
            payload_json=payload,
        )
        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                get_report_scoped=AsyncMock(return_value=report)
            ),
            runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)),
            turns=SimpleNamespace(
                list_finding_blocks_for_runs=AsyncMock(
                    return_value={ops_run_id: compilation.model_dump(mode="json")}
                )
            ),
        )

        class UnitOfWorkContext:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        service = AIOpsRuntimeService(
            uow_factory=lambda: UnitOfWorkContext(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        presentation = asyncio.run(
            service.get_report_presentation(report_id=report_id, domain_id=8)
        )
        briefing = presentation["leadership_briefing"]
        self.assertEqual("HIGH", briefing["risk_level"])
        self.assertIn("业务事务可能被拉长等待", briefing["business_impact"][0])
        dump = _dump(briefing)
        self.assertNotIn("sql_id", dump)
        self.assertNotIn("99", dump)
        uow.turns.list_finding_blocks_for_runs.assert_awaited_once_with(
            ops_run_ids=(ops_run_id,)
        )


if __name__ == "__main__":
    unittest.main()
