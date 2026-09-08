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

    def test_inspection_markdown_facts_are_not_serialized_as_dicts(self) -> None:
        template = SYSTEM_REPORT_TEMPLATES["system:inspection.daily"]
        presentation = report_presentation(
            template=template,
            payload={
                "summary": "巡检已完成",
                "facts": [{
                    "kind": "agent_health_inspection",
                    "markdown": "# 巡检报告\n\n## 主要发现\n\n### 表空间容量临界\n\n使用率达到 **96.88%**。",
                }],
                "gaps": (),
            },
        )
        findings = next(item for item in presentation["sections"] if item["kind"] == "FINDINGS")
        self.assertEqual(
            ["【主要发现】", "【表空间容量临界】", "使用率达到 96.88%。"],
            findings["items"],
        )

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

    def test_template_inspection_report_is_reusable_for_periodic_rollup(self) -> None:
        source = normalize_report_source(
            schema_version="REPORT_CONTENT.v1",
            source_kind="INSPECTION",
            payload={
                "status": "READY",
                "summary": "本期按冻结巡检模板完成 2/2 项检查。",
                "scope": {"inspection_coverage": "已完成 2/2 项检查"},
                "facts": [
                    {"summary": "实例性能指标：检查已完成"},
                    {"summary": "近期告警日志：检查已完成，采集 0 条观测"},
                ],
                "gaps": [],
                "recommendations": ["继续按既定周期执行巡检。"],
                "evidence_refs": [],
            },
        )
        self.assertEqual("READY", source["status"])
        self.assertEqual("已完成 2/2 项检查", source["inspection_coverage"])
        self.assertEqual(2, len(source["facts"]))
        self.assertEqual(
            ("继续按既定周期执行巡检。",),
            source["solution"]["long_term_remediations"],
        )

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

    def test_conversation_report_aggregates_all_completed_turns(self) -> None:
        first_turn, second_turn = uuid7(), uuid7()
        first_run, second_run = uuid7(), uuid7()
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(), blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        result = service._aggregate_conversation_sources(
            conversation=SimpleNamespace(
                conversation_id=uuid7(), title="生产库锁等待排查"
            ),
            source_rows=[
                (
                    SimpleNamespace(turn_id=first_turn, turn_no=1),
                    SimpleNamespace(
                        ops_run_id=first_run,
                        completed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                        created_at=datetime(2026, 9, 1, tzinfo=UTC),
                        plan_snapshot_json={"diagnosis": {"question_summary": "确认锁等待"}},
                    ),
                    SimpleNamespace(
                        artifact_id=uuid7(), content_hash="a" * 64,
                        schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                        payload_json={
                            "status": "READY",
                            "root_cause": {"effective_level": "POSSIBLE"},
                            "diagnosis_rationale": "锁等待来自未提交事务。",
                            "facts": [{"summary": "锁等待持续升高"}],
                            "solution": {"immediate_mitigations": ["确认阻塞会话"]},
                        },
                    ),
                ),
                (
                    SimpleNamespace(turn_id=second_turn, turn_no=2),
                    SimpleNamespace(
                        ops_run_id=second_run,
                        completed_at=datetime(2026, 9, 1, 2, tzinfo=UTC),
                        created_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                        plan_snapshot_json={"diagnosis": {"question_summary": "验证根因"}},
                    ),
                    SimpleNamespace(
                        artifact_id=uuid7(), content_hash="b" * 64,
                        schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                        payload_json={
                            "status": "READY",
                            "root_cause": {"effective_level": "CONFIRMED"},
                            "diagnosis_rationale": "阻塞会话已定位。",
                            "facts": [{"summary": "阻塞会话已确认"}],
                            "solution": {"long_term_remediations": ["优化事务边界"]},
                        },
                    ),
                ),
            ],
            missing_turns=[{"code": "MISSING_FINAL_RESULT", "turn_no": 3}],
        )
        self.assertEqual("PARTIAL", result["status"])
        self.assertEqual("CONFIRMED", result["root_cause"]["effective_level"])
        self.assertEqual(4, len(result["facts"]))
        self.assertEqual(2, len(result["evidence_refs"]))
        self.assertEqual(3, result["conversation"]["turn_count"])
        self.assertEqual(
            ["确认锁等待", "验证根因"],
            result["conversation"]["question_summaries"],
        )

    def test_generate_conversation_report_freezes_every_completed_turn(self) -> None:
        conversation_id, first_turn, second_turn = uuid7(), uuid7(), uuid7()
        first_run, second_run = uuid7(), uuid7()
        first_artifact, second_artifact = uuid7(), uuid7()
        now = datetime(2026, 9, 1, 3, tzinfo=UTC)
        conversation = SimpleNamespace(
            conversation_id=conversation_id, created_by="operator-1",
            title="生产库锁等待排查", agent_id=uuid7(),
        )
        turns = [
            SimpleNamespace(turn_id=first_turn, turn_no=1, status="COMPLETED"),
            SimpleNamespace(turn_id=second_turn, turn_no=2, status="PARTIAL"),
        ]
        runs = {
            first_turn: SimpleNamespace(
                ops_run_id=first_run, status="COMPLETED", final_artifact_id=first_artifact,
                created_at=datetime(2026, 9, 1, tzinfo=UTC), started_at=None,
                completed_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
                plan_snapshot_json={"diagnosis": {"question_summary": "确认锁等待"}},
            ),
            second_turn: SimpleNamespace(
                ops_run_id=second_run, status="PARTIAL", final_artifact_id=second_artifact,
                created_at=datetime(2026, 9, 1, 1, tzinfo=UTC), started_at=None,
                completed_at=datetime(2026, 9, 1, 2, tzinfo=UTC),
                plan_snapshot_json={"diagnosis": {"question_summary": "验证根因"}},
            ),
        }
        artifacts = {
            first_artifact: SimpleNamespace(
                artifact_id=first_artifact, content_hash="a" * 64,
                schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                payload_json={"status": "READY", "diagnosis_rationale": "锁等待来自未提交事务。", "facts": [{"summary": "锁等待持续升高"}]},
            ),
            second_artifact: SimpleNamespace(
                artifact_id=second_artifact, content_hash="b" * 64,
                schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                payload_json={"status": "PARTIAL", "diagnosis_rationale": "阻塞会话已定位。", "facts": [{"summary": "阻塞会话已确认"}]},
            ),
        }
        uow = SimpleNamespace(
            inspections=SimpleNamespace(),
            conversations=SimpleNamespace(get_conversation=AsyncMock(return_value=conversation)),
            turns=SimpleNamespace(
                list_all_turns=AsyncMock(return_value=turns),
                get_run_link=AsyncMock(side_effect=lambda *, turn_id, purpose: SimpleNamespace(ops_run_id=runs[turn_id].ops_run_id)),
            ),
            runs=SimpleNamespace(
                get_run_scoped=AsyncMock(side_effect=lambda *, ops_run_id, domain_id, lock: next(run for run in runs.values() if run.ops_run_id == ops_run_id)),
                get_artifact=AsyncMock(side_effect=lambda *, artifact_id: artifacts[artifact_id]),
                list_tasks=AsyncMock(return_value=[SimpleNamespace(output_artifact_id=second_artifact)]),
                database_now=AsyncMock(return_value=now),
            ),
            commit=AsyncMock(),
        )

        class UnitOfWorkContext:
            async def __aenter__(self): return uow
            async def __aexit__(self, exc_type, exc, traceback): return None

        service = AIOpsRuntimeService(
            uow_factory=lambda: UnitOfWorkContext(), blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        service._publish_diagnosis_report = AsyncMock(return_value=SimpleNamespace(report_id=uuid7()))
        asyncio.run(service.generate_conversation_report(
            domain_id=8, actor_id="operator-1", conversation_id=conversation_id,
            template=SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"], trace_id="trace-session-report",
        ))
        source_override = service._publish_diagnosis_report.await_args.kwargs["source_override"]
        self.assertEqual(4, len(source_override["facts"]))
        self.assertEqual(2, len(source_override["evidence_refs"]))
        self.assertEqual("PARTIAL", source_override["status"])
        uow.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
