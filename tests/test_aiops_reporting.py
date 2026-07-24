"""步骤 10B 不可变巡检报告发布测试。"""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiops_agent.application.runtime.service import AIOpsRuntimeService
from platform_core.identity import uuid7


class InspectionReportPublishingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
