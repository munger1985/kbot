"""步骤 10B 不可变巡检报告发布测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy import Text

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.application.reporting import resolve_system_template
from aiops_agent.contracts.change import ActionVerification
from aiops_agent.contracts.report import ComparisonPlan
from aiops_agent.entities import ReportEntity
from platform_core.identity import uuid7


class InspectionReportPublishingTest(unittest.TestCase):
    def test_report_summary_uses_clob_mapping(self) -> None:
        self.assertIsInstance(ReportEntity.__table__.c.summary.type, Text)

    def test_scheduled_agent_turn_publishes_custom_report(self) -> None:
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            actor_id="system:inspection-scheduler",
            target_id=uuid7(),
            inspection_fire_id=uuid7(),
            plan_snapshot_json={
                "target": {"security_level": 2},
                "client_metadata": {
                    "inspection": {
                        "template_id": "database_custom",
                        "template_version": "1.0.0",
                        "schedule_type": "CRON",
                        "timezone": "Asia/Shanghai",
                        "period_start": "2026-07-22T16:00:00+00:00",
                        "period_end": "2026-07-23T16:00:00+00:00",
                    }
                },
            },
        )
        source = SimpleNamespace(
            artifact_id=uuid7(),
            schema_version="AIOPS_TURN_RESULT.v1",
            content_hash="c" * 64,
            payload_json={
                "schema_version": "AIOPS_TURN_RESULT.v1",
                "status": "COMPLETED",
                "sufficiency_status": "ANSWERABLE",
                "blocks": [
                    {
                        "block_type": "MARKDOWN",
                        "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                        "payload": {"markdown": "数据库整体健康。"},
                        "evidence_refs": [],
                    }
                ],
            },
        )

        async def add_artifact(entity):
            entity.artifact_id = uuid7()
            return entity

        async def publish_report(entity):
            entity.is_current = 1
            return entity

        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                publish_report=AsyncMock(side_effect=publish_report),
                add_report_sources=AsyncMock(),
            ),
            runs=SimpleNamespace(
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_report_ready=AsyncMock()
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )

        result = asyncio.run(
            service._publish_turn_inspection_report(
                uow=uow,
                run=run,
                task=SimpleNamespace(ops_task_id=uuid7()),
                source_artifact=source,
                now=datetime(2026, 7, 24, tzinfo=UTC),
                trace_id="trace-agent-inspection",
            )
        )

        self.assertEqual(result.schema_version, "REPORT_CONTENT.v1")
        self.assertEqual(
            result.payload_json["report_type"], "INSPECTION_CUSTOM"
        )
        self.assertEqual(
            result.payload_json["facts"][0]["markdown"],
            "数据库整体健康。",
        )

    def test_scheduled_agent_turn_keeps_long_chinese_summary(self) -> None:
        markdown = "数据库健康检查结论。" * 600
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            actor_id="system:inspection-scheduler",
            target_id=uuid7(),
            inspection_fire_id=uuid7(),
            plan_snapshot_json={
                "target": {"security_level": 2},
                "client_metadata": {
                    "inspection": {
                        "template_id": "database_daily",
                        "template_version": "1.0.0",
                        "schedule_type": "DAILY",
                        "timezone": "Asia/Shanghai",
                        "period_start": "2026-07-22T16:00:00+00:00",
                        "period_end": "2026-07-23T16:00:00+00:00",
                    }
                },
            },
        )
        source = SimpleNamespace(
            artifact_id=uuid7(),
            schema_version="AIOPS_TURN_RESULT.v1",
            content_hash="d" * 64,
            payload_json={
                "status": "COMPLETED",
                "sufficiency_status": "ANSWERABLE",
                "blocks": [
                    {
                        "block_type": "MARKDOWN",
                        "schema_version": "AIOPS_MARKDOWN_BLOCK.v1",
                        "payload": {"markdown": markdown},
                        "evidence_refs": [],
                    }
                ],
            },
        )

        async def add_artifact(entity):
            entity.artifact_id = uuid7()
            return entity

        async def publish_report(entity):
            entity.is_current = 1
            return entity

        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                publish_report=AsyncMock(side_effect=publish_report),
                add_report_sources=AsyncMock(),
            ),
            runs=SimpleNamespace(
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_report_ready=AsyncMock()
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )

        asyncio.run(
            service._publish_turn_inspection_report(
                uow=uow,
                run=run,
                task=SimpleNamespace(ops_task_id=uuid7()),
                source_artifact=source,
                now=datetime(2026, 7, 24, tzinfo=UTC),
                trace_id="trace-long-agent-inspection",
            )
        )

        report = uow.inspections.publish_report.await_args.args[0]
        self.assertGreater(len(markdown.encode("utf-8")), 4000)
        self.assertEqual(markdown, report.summary)

    def test_schedule_result_publishes_report_content_and_projection(
        self,
    ) -> None:
        run_id = uuid7()
        task_id = uuid7()
        target_id = uuid7()
        fire_id = uuid7()
        source_id = uuid7()
        report_artifact_id = uuid7()
        run = SimpleNamespace(
            ops_run_id=run_id,
            actor_id="portal:user-1",
            target_id=target_id,
            inspection_fire_id=fire_id,
            plan_snapshot_json={
                "target": {"security_level": 3},
                "client_metadata": {
                    "inspection": {
                        "template_id": "database_daily",
                        "template_version": "1.0.0",
                        "schedule_type": "DAILY",
                        "timezone": "Asia/Shanghai",
                        "period_start": (
                            "2026-07-22T16:00:00+00:00"
                        ),
                        "period_end": (
                            "2026-07-23T16:00:00+00:00"
                        ),
                    }
                },
            },
        )
        task = SimpleNamespace(ops_task_id=task_id)
        source = SimpleNamespace(
            artifact_id=source_id,
            schema_version="DB_DIAGNOSTIC_REPORT.v1",
            content_hash="a" * 64,
            payload_json={
                "status": "PARTIAL",
                "observation_count": 4,
                "gap_count": 1,
                "tools": ["db.instance.identity"],
                "gaps": [
                    {
                        "code": "CAPABILITY_UNAVAILABLE",
                        "tool_id": "db.replication.status",
                        "detail": "当前目标不支持复制检查",
                        "retryable": False,
                    }
                ],
                "provenance": {
                    "catalog_hash": "b" * 64,
                    "deterministic": True,
                    "llm_used": False,
                },
            },
        )

        async def add_artifact(entity):
            entity.artifact_id = report_artifact_id
            return entity

        async def publish_report(entity):
            entity.is_current = 1
            return entity

        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                publish_report=AsyncMock(side_effect=publish_report)
            ),
            runs=SimpleNamespace(
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_report_ready=AsyncMock()
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        result = asyncio.run(
            service._publish_inspection_report(
                uow=uow,
                run=run,
                task=task,
                source_artifact=source,
                now=datetime(2026, 7, 24, tzinfo=UTC),
                trace_id="trace-report",
            )
        )
        self.assertEqual(result.artifact_id, report_artifact_id)
        self.assertEqual(result.schema_version, "REPORT_CONTENT.v1")
        self.assertEqual(result.payload_json["status"], "PARTIAL")
        report = (
            uow.inspections.publish_report.await_args.args[0]
        )
        self.assertEqual(report.report_type, "INSPECTION_DAILY")
        self.assertEqual(report.is_current, 1)
        self.assertEqual(
            report.content_artifact_id, report_artifact_id
        )
        uow.outbox.add.assert_awaited_once()


class DiagnosisReportPublishingTest(unittest.TestCase):
    def test_formal_diagnosis_is_published_without_replacing_chat_artifact(
        self,
    ) -> None:
        run_id = uuid7()
        target_id = uuid7()
        task_id = uuid7()
        source_id = uuid7()
        created_at = datetime(2026, 7, 24, 1, tzinfo=UTC)
        run = SimpleNamespace(
            ops_run_id=run_id,
            domain_id=8,
            actor_id="portal:user-1",
            target_id=target_id,
            created_at=created_at,
            plan_snapshot_json={
                "diagnosis": {
                    "question_summary": "分析数据库响应变慢"
                },
                "effective_capabilities": {
                    "monitor_read": True,
                    "database_read": True,
                    "mutation_execute": False,
                },
            },
        )
        task = SimpleNamespace(ops_task_id=task_id)
        source = SimpleNamespace(
            artifact_id=source_id,
            schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
            content_hash="a" * 64,
            payload_json={
                "status": "READY",
                "output_kind": "DIAGNOSIS_REPORT",
                "report_decision_reasons": ["ISSUE_DETECTED"],
                "root_cause": {"effective_level": "PROBABLE"},
                "diagnosis_rationale": "锁等待导致响应时间升高",
                "facts": [
                    {
                        "fact_id": "fact-1",
                        "fact_summary": "锁等待持续升高",
                        "trust_level": "SOURCE_VERIFIED",
                    }
                ],
                "gaps": [],
                "solution": {
                    "immediate_mitigations": ["确认阻塞源"],
                    "long_term_remediations": ["优化事务边界"],
                },
                "model_receipt_hashes": ["b" * 64],
            },
        )

        async def add_artifact(entity):
            entity.artifact_id = uuid7()
            return entity

        async def publish_report(entity):
            entity.is_current = 1
            return entity

        uow = SimpleNamespace(
            inspections=SimpleNamespace(
                publish_report=AsyncMock(side_effect=publish_report),
                add_report_sources=AsyncMock(),
            ),
            targets=SimpleNamespace(
                get_scoped=AsyncMock(
                    return_value=SimpleNamespace(security_level=3)
                ),
            ),
            runs=SimpleNamespace(
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_report_ready=AsyncMock()
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        result = asyncio.run(
            service._publish_diagnosis_report(
                uow=uow,
                run=run,
                task=task,
                source_artifact=source,
                now=datetime(2026, 7, 24, 2, tzinfo=UTC),
                trace_id="trace-diagnosis-report",
                template=resolve_system_template("system:diagnosis.standard"),
                actor_id="operator-1",
            )
        )
        self.assertIsNotNone(result)
        report = uow.inspections.publish_report.await_args.args[0]
        self.assertEqual(report.report_type, "INCIDENT")
        self.assertEqual(report.template_id, "system:diagnosis.standard")
        self.assertEqual(report.status, "READY")
        self.assertEqual(report.security_level, 3)
        uow.targets.get_scoped.assert_awaited_once_with(
            target_id=target_id,
            domain_id=8,
        )
        content = uow.runs.add_artifact.await_args.args[0]
        self.assertEqual(content.schema_version, "REPORT_CONTENT.v1")
        self.assertIn("锁等待导致响应时间升高", content.payload_json["summary"])
        uow.inspections.add_report_sources.assert_awaited_once()
        source = uow.inspections.add_report_sources.await_args.args[0][0]
        self.assertEqual(source.ops_run_id, run_id)
        self.assertEqual(source.source_artifact_id, source_id)


class ComparisonReportPublishingTest(unittest.TestCase):
    def test_verified_action_publishes_improved_comparison_report(
        self,
    ) -> None:
        source_run_id = uuid7()
        verification_run_id = uuid7()
        proposal_id = uuid7()
        target_id = uuid7()
        task_id = uuid7()
        source_result_id = uuid7()
        plan_artifact_id = uuid7()
        verification_artifact_id = uuid7()
        created_artifact_ids = iter((uuid7(), uuid7()))
        baseline_start = datetime(2026, 7, 24, 1, tzinfo=UTC)
        baseline_end = datetime(2026, 7, 24, 2, tzinfo=UTC)
        after_start = datetime(2026, 7, 24, 3, tzinfo=UTC)
        after_end = datetime(2026, 7, 24, 4, tzinfo=UTC)
        proposal = SimpleNamespace(
            proposal_id=proposal_id,
            ops_run_id=source_run_id,
            solution_group_key="blocking-session",
            action_template_id="oracle.kill-session",
            action_template_version="1.0.0",
        )
        source_run = SimpleNamespace(
            ops_run_id=source_run_id,
            target_id=target_id,
            actor_id="portal:user-1",
        )
        source_result = SimpleNamespace(
            artifact_id=source_result_id,
            payload_json={"execution_id": str(uuid7())},
        )
        plan = ComparisonPlan(
            proposal_id=str(proposal_id),
            source_run_id=str(source_run_id),
            solution_group_key="blocking-session",
            action_template_id="oracle.kill-session",
            action_template_version="1.0.0",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            settle_delay_seconds=0,
            after_window_seconds=3600,
            primary_signals=("target_absent", "blocking_absent"),
            required_tool_refs=(
                "db.session.active",
                "db.session.blocking_chain",
            ),
            baseline_evidence_refs=("artifact:baseline",),
        )
        plan_artifact = SimpleNamespace(
            artifact_id=plan_artifact_id,
            schema_version="COMPARISON_PLAN.v1",
            payload_json=plan.model_dump(mode="json"),
            content_hash="a" * 64,
        )
        verification = ActionVerification(
            proposal_id=str(proposal_id),
            source_run_id=str(source_run_id),
            result_artifact_id=str(source_result_id),
            status="VERIFIED",
            summary="动作效果已验证",
            target_still_present=False,
            blocking_still_present=False,
            checked_tool_refs=(
                "db.session.active",
                "db.session.blocking_chain",
            ),
            evidence_hashes=("b" * 64,),
        )
        verification_artifact = SimpleNamespace(
            artifact_id=verification_artifact_id,
            schema_version="ACTION_VERIFICATION.v1",
            payload_json=verification.model_dump(mode="json"),
            content_hash="c" * 64,
        )
        verification_run = SimpleNamespace(
            ops_run_id=verification_run_id,
            target_id=target_id,
            source_proposal_id=proposal_id,
            source_result_artifact_id=source_result_id,
            created_at=after_start,
            plan_snapshot_json={"target": {"security_level": 4}},
        )
        task = SimpleNamespace(ops_task_id=task_id)

        async def add_artifact(entity):
            entity.artifact_id = next(created_artifact_ids)
            return entity

        async def publish_report(entity):
            entity.is_current = 1
            return entity

        uow = SimpleNamespace(
            changes=SimpleNamespace(
                get_proposal=AsyncMock(return_value=proposal)
            ),
            inspections=SimpleNamespace(
                publish_report=AsyncMock(side_effect=publish_report)
            ),
            runs=SimpleNamespace(
                get_run=AsyncMock(return_value=source_run),
                get_artifact=AsyncMock(return_value=source_result),
                get_artifact_by_key=AsyncMock(
                    return_value=plan_artifact
                ),
                list_tasks=AsyncMock(return_value=[]),
                add_artifact=AsyncMock(side_effect=add_artifact),
                append_event=AsyncMock(),
            ),
            outbox=SimpleNamespace(
                add=AsyncMock(side_effect=lambda entity: entity)
            ),
            platform_notifications=SimpleNamespace(
                emit_report_ready=AsyncMock()
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=AsyncMock(),
            blueprint_registry=AsyncMock(),
            handler_registry=AsyncMock(),
        )
        result = asyncio.run(
            service._publish_comparison_report(
                uow=uow,
                run=verification_run,
                task=task,
                verification_artifact=verification_artifact,
                now=after_end,
                trace_id="trace-comparison",
            )
        )
        self.assertEqual(result.schema_version, "REPORT_CONTENT.v1")
        report = (
            uow.inspections.publish_report.await_args.args[0]
        )
        self.assertEqual(report.ops_run_id, source_run_id)
        self.assertEqual(report.report_type, "COMPARISON")
        self.assertEqual(report.result, "RESOLVED")
        self.assertEqual(report.baseline_start, baseline_start)
        self.assertEqual(report.after_end, after_end)
        comparison_artifact = (
            uow.runs.add_artifact.await_args_list[0].args[0]
        )
        self.assertEqual(
            comparison_artifact.schema_version, "COMPARISON_RESULT.v1"
        )
        self.assertEqual(
            comparison_artifact.payload_json["result"], "RESOLVED"
        )

    def test_evidence_gap_forces_inconclusive_result(self) -> None:
        verification = ActionVerification(
            proposal_id=str(uuid7()),
            source_run_id=str(uuid7()),
            result_artifact_id=str(uuid7()),
            status="INCONCLUSIVE",
            summary="证据不足",
            gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
        )
        result, rationale = AIOpsRuntimeService._comparison_result(
            verification
        )
        self.assertEqual(result, "INCONCLUSIVE")
        self.assertEqual(rationale, ("EVIDENCE_NOT_COMPARABLE",))


if __name__ == "__main__":
    unittest.main()
