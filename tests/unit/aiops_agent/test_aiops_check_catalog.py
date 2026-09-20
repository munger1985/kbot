"""巡检 Check Catalog 加载、默认勾选与计划快照校验。"""

from __future__ import annotations

import hashlib
import json
import unittest

from pydantic import ValidationError

from aiops_agent.application.inspections.check_catalog import (
    _CATALOG_PATH,
    _canonical_json,
    compile_selected_check_steps,
    default_selected_check_ids,
    load_check_catalog,
    normalize_selected_check_ids,
    selected_check_ids_from_json,
)
from platform_core.contracts.aiops import InspectionPlanCreate
from platform_core.identity import uuid7


class CheckCatalogLoaderTest(unittest.TestCase):
    def test_catalog_loads_ready_and_planned_groups(self) -> None:
        catalog = load_check_catalog()
        payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        items = [item for group in catalog.groups for item in group.checks]
        ready = [item for item in items if item.availability == "READY"]
        planned = [item for item in items if item.availability == "PLANNED"]

        self.assertEqual(7, len(catalog.groups))
        self.assertEqual(24, len(items))
        self.assertEqual(19, len(ready))
        self.assertEqual(5, len(planned))
        self.assertEqual(
            hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
            catalog.catalog_hash,
        )
        for item in ready:
            self.assertTrue(item.tool_id or item.playbook_id)

    def test_default_selection_uses_ready_daily_or_weekly(self) -> None:
        daily = default_selected_check_ids("DAILY")
        weekly = default_selected_check_ids("WEEKLY")
        cron = default_selected_check_ids("CRON")
        catalog = load_check_catalog()
        items = [item for group in catalog.groups for item in group.checks]
        self.assertEqual(
            tuple(
                item.check_id
                for item in items
                if item.availability == "READY"
                and "DAILY" in item.default_for
                and item.tool_id
                and not item.tool_id.startswith("user.")
            ),
            daily,
        )
        self.assertEqual(
            tuple(
                item.check_id
                for item in items
                if item.availability == "READY"
                and "WEEKLY" in item.default_for
                and item.tool_id
                and not item.tool_id.startswith("user.")
            ),
            weekly,
        )
        self.assertEqual(daily, cron)
        self.assertTrue(daily)
        self.assertNotIn("oracle.storage.row_chaining", daily)
        self.assertIn("oracle.session.long_transaction", daily)
        self.assertNotIn("oracle.exadata.health_report", daily)
        self.assertNotIn("oracle.exadata.health_report", weekly)

    def test_normalize_rejects_empty_unknown_and_planned(self) -> None:
        daily = default_selected_check_ids("DAILY")
        with self.assertRaisesRegex(ValueError, "至少勾选一个检查项"):
            normalize_selected_check_ids([])
        with self.assertRaisesRegex(ValueError, "未知检查项"):
            normalize_selected_check_ids(["oracle.not.a.check"])
        with self.assertRaisesRegex(ValueError, "检查项尚未开放"):
            normalize_selected_check_ids(["oracle.storage.row_chaining"])
        with self.assertRaisesRegex(ValueError, "检查项必须是 ID 列表"):
            normalize_selected_check_ids("oracle.backup.rman_volume")

        unordered = (daily[-1], daily[0], daily[0])
        self.assertEqual(
            (daily[0], daily[-1]),
            normalize_selected_check_ids(unordered),
        )

    def test_snapshot_json_only_accepts_id_list(self) -> None:
        daily = default_selected_check_ids("DAILY")
        self.assertEqual(daily, selected_check_ids_from_json(list(daily)))
        with self.assertRaisesRegex(ValueError, "检查项快照必须是 ID 列表"):
            selected_check_ids_from_json({"ids": list(daily)})

    def test_create_contract_requires_selected_check_ids(self) -> None:
        payload = {
            "display_name": "每日巡检",
            "agent_id": str(uuid7()),
            "schedule_type": "DAILY",
            "cron_expression": "0 13 * * *",
            "timezone": "Asia/Shanghai",
            "template_id": "database_daily",
            "template_version": "1.0.0",
            "timeout_seconds": 1800,
            "schedule_resolver_version": "1.0.0",
        }
        with self.assertRaises(ValidationError):
            InspectionPlanCreate(**payload)
        created = InspectionPlanCreate(
            **payload,
            selected_check_ids=default_selected_check_ids("DAILY"),
        )
        self.assertEqual(
            default_selected_check_ids("DAILY"),
            created.selected_check_ids,
        )


    def test_exadata_health_report_can_bind_but_does_not_execute(self) -> None:
        selected = normalize_selected_check_ids(
            ["oracle.session.lock_wait", "oracle.exadata.health_report"]
        )
        self.assertIn("oracle.exadata.health_report", selected)
        steps = compile_selected_check_steps(selected)
        self.assertEqual(["db.session.blocking_chain"], [item["tool_id"] for item in steps])
        with self.assertRaisesRegex(ValueError, "勾选检查项没有可执行的取证工具"):
            compile_selected_check_steps(["oracle.exadata.health_report"])

    def test_compile_selected_check_steps_dedupes_tools(self) -> None:
        daily = default_selected_check_ids("DAILY")
        steps = compile_selected_check_steps(daily)
        tool_ids = [item["tool_id"] for item in steps]
        self.assertEqual(len(tool_ids), len(set(tool_ids)))
        self.assertIn("db.session.blocking_chain", tool_ids)
        self.assertIn("db.storage.capacity", tool_ids)
        self.assertNotIn("db.alert.recent", tool_ids)
        self.assertNotIn("db.scheduler.failed_jobs", tool_ids)
        backup = next(
            item for item in steps if item["tool_id"] == "db.backup.recent_jobs"
        )
        self.assertGreaterEqual(len(backup["check_ids"]), 2)
        self.assertIn("、", backup["title"])
        subset = compile_selected_check_steps(
            ["oracle.session.lock_wait", "oracle.storage.tablespace_headroom"]
        )
        self.assertEqual(
            ["db.storage.capacity", "db.session.blocking_chain"],
            [item["tool_id"] for item in subset],
        )
        self.assertEqual(["LOCK_WAIT"], [item["expected_evidence_kind"] for item in subset[1:]])
        self.assertTrue(all(
            item["measurement_semantics"] == "CURRENT_ACTIVITY"
            for item in steps
        ))

    def test_compile_weekly_trend_required_uses_historical_samples(self) -> None:
        selected = [
            "oracle.session.lock_wait",
            "oracle.storage.tablespace_headroom",
            "oracle.storage.growth",
        ]
        daily = compile_selected_check_steps(selected, schedule_type="DAILY")
        weekly = compile_selected_check_steps(selected, schedule_type="WEEKLY")
        cron = compile_selected_check_steps(selected, schedule_type="CRON")
        daily_by_tool = {item["tool_id"]: item for item in daily}
        weekly_by_tool = {item["tool_id"]: item for item in weekly}
        cron_by_tool = {item["tool_id"]: item for item in cron}

        self.assertEqual("CURRENT_ACTIVITY", daily_by_tool["db.storage.capacity"]["measurement_semantics"])
        self.assertTrue(daily_by_tool["db.storage.capacity"]["trend_required"])
        self.assertEqual("CURRENT_ACTIVITY", daily_by_tool["db.session.blocking_chain"]["measurement_semantics"])
        self.assertFalse(daily_by_tool["db.session.blocking_chain"]["trend_required"])

        self.assertEqual("HISTORICAL_SAMPLES", weekly_by_tool["db.storage.capacity"]["measurement_semantics"])
        self.assertTrue(weekly_by_tool["db.storage.capacity"]["trend_required"])
        self.assertEqual(
            ["oracle.storage.tablespace_headroom", "oracle.storage.growth"],
            weekly_by_tool["db.storage.capacity"]["check_ids"],
        )
        self.assertEqual("CURRENT_ACTIVITY", weekly_by_tool["db.session.blocking_chain"]["measurement_semantics"])
        self.assertFalse(weekly_by_tool["db.session.blocking_chain"]["trend_required"])
        self.assertEqual(
            cron_by_tool["db.storage.capacity"]["measurement_semantics"],
            daily_by_tool["db.storage.capacity"]["measurement_semantics"],
        )


if __name__ == "__main__":
    unittest.main()
