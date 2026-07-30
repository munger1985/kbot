"""Step 10D Fire 与 Report 查询面的作用域和分页测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.runtime.service import AIOpsRuntimeService
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
