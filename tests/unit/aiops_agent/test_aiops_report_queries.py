"""Step 10D Fire 与 Report 查询面的作用域和分页测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.contracts.report import ReportContent
from aiops_agent.application.reporting import SYSTEM_REPORT_TEMPLATES
from platform_core.contracts.aiops import ReportSectionEdit
from platform_core.identity import uuid7


def _scope() -> ConfigurationScope:
    return ConfigurationScope(
        domain_id=200,
        principal_id="API_CLIENT:portal",
        actor_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
    )


def _context(uow):
    context = AsyncMock()
    context.__aenter__.return_value = uow
    context.__aexit__.return_value = None
    return context


class ReportQueryTest(unittest.TestCase):
    def test_current_report_page_uses_scoped_keyset_cursor(self) -> None:
        now = datetime(2026, 7, 24, tzinfo=UTC)

        def report(offset: int):
            return SimpleNamespace(
                report_id=uuid7(),
                report_key=f"inspection.daily.{offset}",
                report_type="INSPECTION_DAILY",
                report_version=1,
                title="日常巡检报告",
                status="READY",
                target_id=uuid7(),
                period_start=now - timedelta(days=1),
                period_end=now,
                summary="巡检完成",
                created_at=now - timedelta(seconds=offset),
            )

        entities = [report(1), report(2)]
        inspections = SimpleNamespace(
            page_current_reports=AsyncMock(return_value=entities)
        )
        uow = SimpleNamespace(inspections=inspections)
        cursor_codec = Mock()
        cursor_codec.encode.return_value = "signed-next"
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(uow),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
            cursor_codec=cursor_codec,
        )
        result = asyncio.run(
            service.list_reports(
                scope=_scope(),
                target_id=None,
                report_type="INSPECTION_DAILY",
                cursor=None,
                limit=1,
            )
        )
        self.assertEqual(len(result.items), 1)
        self.assertTrue(result.has_more)
        self.assertEqual(result.next_cursor, "signed-next")
        kwargs = inspections.page_current_reports.await_args.kwargs
        self.assertEqual(kwargs["domain_id"], 200)
        self.assertEqual(kwargs["limit"], 2)
        cursor_filters = cursor_codec.encode.call_args.kwargs["filters"]
        self.assertEqual(
            cursor_filters,
            {"target_id": None, "report_type": "INSPECTION_DAILY"},
        )

    def test_report_versions_anchor_is_domain_scoped(self) -> None:
        report_id = uuid7()
        source_run_id = uuid7()
        now = datetime(2026, 7, 24, tzinfo=UTC)
        anchor = SimpleNamespace(
            report_id=report_id,
            ops_run_id=source_run_id,
            report_key="comparison.action.example",
        )
        version = SimpleNamespace(
            report_id=report_id,
            report_version=2,
            status="READY",
            created_at=now,
        )
        inspections = SimpleNamespace(
            get_report_scoped=AsyncMock(return_value=anchor),
            page_report_versions=AsyncMock(return_value=[version]),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(
                SimpleNamespace(inspections=inspections)
            ),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
            cursor_codec=Mock(),
        )
        result = asyncio.run(
            service.list_report_versions(
                scope=_scope(),
                report_id=report_id,
                cursor=None,
                limit=50,
            )
        )
        self.assertEqual(result.items[0].report_version, 2)
        anchor_scope = inspections.get_report_scoped.await_args.kwargs
        self.assertEqual(anchor_scope["domain_id"], 200)
        page_scope = inspections.page_report_versions.await_args.kwargs
        self.assertEqual(page_scope["ops_run_id"], source_run_id)
        self.assertEqual(
            page_scope["report_key"], "comparison.action.example"
        )

    def test_edit_report_creates_new_version_without_changing_evidence(self) -> None:
        now = datetime(2026, 9, 4, tzinfo=UTC)
        report_id, artifact_id, task_id, target_id, run_id = (
            uuid7(), uuid7(), uuid7(), uuid7(), uuid7()
        )
        template = SYSTEM_REPORT_TEMPLATES["system:diagnosis.standard"]
        content = ReportContent(
            report_key="diagnosis.standard",
            report_type="INCIDENT",
            ops_run_id=str(run_id),
            target_id=str(target_id),
            title="原报告",
            status="READY",
            summary="原摘要",
            period_start=now - timedelta(hours=1),
            period_end=now,
            scope={},
            evidence_refs=({"artifact_id": "evidence-1", "content_hash": "b" * 64},),
            provenance={"template": {
                "template_ref": template.template_ref,
                "version": template.version,
                "definition": template.definition,
            }},
        )
        report = SimpleNamespace(
            report_id=report_id, ops_run_id=run_id, target_id=target_id,
            report_key="diagnosis.standard", report_version=1, is_current=1,
            report_type="INCIDENT", title="原报告", status="READY",
            period_start=now - timedelta(hours=1), period_end=now,
            baseline_start=None, baseline_end=None, after_start=None,
            after_end=None, result=None, template_id=template.template_ref,
            template_version=template.version, generated_by_task_id=task_id,
            content_artifact_id=artifact_id, content_hash="a" * 64,
            summary="原摘要", security_level=1, created_at=now,
        )
        source_artifact = SimpleNamespace(
            artifact_id=artifact_id, schema_version="REPORT_CONTENT.v1",
            content_hash="a" * 64, payload_json=content.model_dump(mode="json"),
            artifact_type="REPORT_CONTENT",
        )
        saved_artifact = SimpleNamespace(
            artifact_id=uuid7(), artifact_type="REPORT_CONTENT",
            schema_version="REPORT_CONTENT.v1", content_hash="c" * 64,
        )
        inspections = SimpleNamespace(
            get_report_scoped=AsyncMock(return_value=report),
            publish_report=AsyncMock(side_effect=lambda entity: entity),
            list_report_sources=AsyncMock(return_value=[]),
            add_report_sources=AsyncMock(),
        )
        runs = SimpleNamespace(
            get_artifact=AsyncMock(return_value=source_artifact),
            database_now=AsyncMock(return_value=now),
            add_artifact=AsyncMock(return_value=saved_artifact),
            append_event=AsyncMock(),
        )
        uow = SimpleNamespace(
            inspections=inspections, runs=runs,
            outbox=SimpleNamespace(add=AsyncMock()), commit=AsyncMock(),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(uow), blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        result = asyncio.run(service.edit_report(
            report_id=report_id, domain_id=200, actor_id="user-1",
            expected_report_version=1, title="修订报告",
            sections=(ReportSectionEdit(
                kind="EXECUTIVE_SUMMARY", items=("修订摘要",),
            ),),
            trace_id="trace-1",
        ))
        self.assertEqual(result.report_version, 2)
        self.assertEqual(result.title, "修订报告")
        self.assertEqual(result.corrected_from_report_id, report_id)
        published = inspections.publish_report.await_args.args[0]
        self.assertEqual(published.supersedes_report_id, report_id)
        artifact_payload = runs.add_artifact.await_args.args[0].payload_json
        self.assertEqual(artifact_payload["evidence_refs"], content.model_dump(mode="json")["evidence_refs"])
        self.assertEqual(
            artifact_payload["presentation_overrides"]["EXECUTIVE_SUMMARY"],
            ["修订摘要"],
        )


class InspectionFireQueryTest(unittest.TestCase):
    def test_fire_detail_returns_only_linked_run_ids(self) -> None:
        fire_id = uuid7()
        plan_id = uuid7()
        now = datetime(2026, 7, 24, tzinfo=UTC)
        fire = SimpleNamespace(
            inspection_fire_id=fire_id,
            inspection_plan_id=plan_id,
            scheduled_for=now,
            status="PARTIAL",
            target_count=2,
            completed_count=1,
            failed_count=1,
            created_at=now,
            completed_at=now + timedelta(minutes=3),
        )
        run_ids = (uuid7(), uuid7())
        inspections = SimpleNamespace(
            get_fire_scoped=AsyncMock(return_value=fire),
            list_runs_for_fire=AsyncMock(
                return_value=[
                    SimpleNamespace(ops_run_id=item) for item in run_ids
                ]
            ),
        )
        service = AIOpsRuntimeService(
            uow_factory=lambda: _context(
                SimpleNamespace(inspections=inspections)
            ),
            blueprint_registry=Mock(),
            handler_registry=Mock(),
        )
        result = asyncio.run(
            service.get_inspection_fire(
                inspection_fire_id=fire_id,
                domain_id=200,
            )
        )
        self.assertEqual(result.plan_id, plan_id)
        self.assertEqual(result.run_ids, run_ids)
        scoped = inspections.get_fire_scoped.await_args.kwargs
        self.assertEqual(scoped["domain_id"], 200)


if __name__ == "__main__":
    unittest.main()
