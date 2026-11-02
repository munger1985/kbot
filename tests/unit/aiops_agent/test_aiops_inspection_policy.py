"""巡检备份策略与证据覆盖的跨数据库回归。"""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from aiops_agent.application.errors import InspectionPreparationError
from aiops_agent.application.inspections.backup_policy import (
    backup_policy_tool_id,
    compile_backup_policy_steps,
)
from aiops_agent.application.inspections.coverage import inspection_evidence_coverage


class InspectionBackupPolicyTest(unittest.TestCase):
    policy = {"enabled": True, "success_max_age_hours": 26}
    tools = {
        "ORACLE": "db.backup.recent_jobs",
        "POSTGRESQL": "db.postgresql.archiver.status",
        "MYSQL": "db.mysql.binlog.status",
    }

    def test_policy_resolves_current_catalog_for_every_database(self) -> None:
        for database_type, tool_id in self.tools.items():
            with self.subTest(database_type=database_type):
                steps = compile_backup_policy_steps(
                    (), database_type=database_type,
                    schedule_type="DAILY", backup_policy=self.policy,
                )
                self.assertEqual(tool_id, backup_policy_tool_id(database_type))
                self.assertEqual([tool_id], [step["tool_id"] for step in steps])
                self.assertTrue(steps[0]["check_ids"])
                self.assertTrue(steps[0]["report_columns"])

    def test_existing_frozen_steps_do_not_recompile_current_catalog(self) -> None:
        for database_type, tool_id in self.tools.items():
            with self.subTest(database_type=database_type):
                frozen = ({
                    "tool_id": tool_id,
                    "title": "冻结的备份取证",
                    "check_ids": ["frozen-check-id"],
                    "input": {"limit": 17},
                },)
                original = deepcopy(frozen)
                with patch(
                    "aiops_agent.application.inspections.backup_policy.load_check_catalog",
                    side_effect=AssertionError("不应重读目录"),
                ):
                    steps = compile_backup_policy_steps(
                        frozen, database_type=database_type,
                        schedule_type="DAILY", backup_policy=self.policy,
                    )
                self.assertEqual(1, len(steps))
                self.assertEqual(["frozen-check-id"], steps[0]["check_ids"])
                self.assertEqual(17, steps[0]["input"]["limit"])
                self.assertEqual(original, frozen)

    def test_oracle_query_window_covers_policy_without_shortening_snapshot(self) -> None:
        for hours, existing_days, expected_days in (
            (26, 0, 7), (240, 0, 11), (3000, 0, 90), (26, 30, 30),
        ):
            with self.subTest(hours=hours, existing_days=existing_days):
                steps = compile_backup_policy_steps(
                    ({"tool_id": self.tools["ORACLE"], "input": {"days": existing_days}},),
                    database_type="ORACLE", schedule_type="DAILY",
                    backup_policy={"enabled": True, "success_max_age_hours": hours},
                )
                self.assertEqual(expected_days, steps[0]["input"]["days"])

    def test_disabled_or_non_daily_policy_does_not_add_steps(self) -> None:
        frozen = ({"tool_id": "db.instance.identity"},)
        for schedule_type, policy in (
            ("DAILY", None), ("DAILY", {"enabled": False}),
            ("WEEKLY", self.policy), ("MONTHLY", self.policy),
        ):
            with self.subTest(schedule_type=schedule_type, policy=policy):
                self.assertIs(
                    frozen,
                    compile_backup_policy_steps(
                        frozen, database_type="POSTGRESQL",
                        schedule_type=schedule_type, backup_policy=policy,
                    ),
                )

    def test_missing_policy_tool_is_not_silently_skipped(self) -> None:
        with patch(
            "aiops_agent.application.inspections.backup_policy.load_check_catalog",
            return_value=SimpleNamespace(groups=()),
        ):
            with self.assertRaisesRegex(ValueError, "没有可执行的检查目录项"):
                compile_backup_policy_steps(
                    (), database_type="POSTGRESQL",
                    schedule_type="DAILY", backup_policy=self.policy,
                )

    def test_preparation_error_has_safe_stage_and_is_not_retryable(self) -> None:
        for stage in ("INSPECTION_TEMPLATE", "INSPECTION_BACKUP_POLICY"):
            with self.subTest(stage=stage):
                error = InspectionPreparationError(stage=stage)
                self.assertEqual(f"AIOPS_{stage}_INVALID", error.code)
                self.assertEqual(stage, error.stage)
                self.assertEqual(error.message, error.safe_detail)
                self.assertFalse(error.retryable)


class InspectionEvidenceCoverageTest(unittest.TestCase):
    def test_one_tool_can_cover_multiple_semantic_checks(self) -> None:
        coverage = inspection_evidence_coverage(
            (
                {"tool_id": "db.storage.capacity", "check_ids": ["space", "files", "growth"],
                 "check_status": "OBSERVED", "evidence_refs": ["artifact:capacity"]},
                {"tool_id": "db.session.blocking_chain", "check_ids": ["locks"],
                 "check_status": "MISSING"},
            ),
            (),
        )
        self.assertEqual(3, coverage["evidence_covered_count"])
        self.assertEqual(4, coverage["evidence_total_count"])
        self.assertEqual(75.0, coverage["evidence_coverage_percent"])
        self.assertEqual(50.0, coverage["tool_execution_coverage_percent"])

    def test_evidence_gap_does_not_erase_successful_tool_execution(self) -> None:
        coverage = inspection_evidence_coverage(
            ({"tool_id": "db.storage.capacity", "check_ids": ["space", "files"],
              "evidence_refs": ["artifact:capacity"]},),
            ({"step_id": "db.storage.capacity", "code": "FINDING_COLUMN_MISSING"},),
        )
        self.assertEqual(0.0, coverage["evidence_coverage_percent"])
        self.assertEqual(100.0, coverage["tool_execution_coverage_percent"])

    def test_incomplete_observation_is_not_complete_evidence(self) -> None:
        for status in ("PARTIAL", "GAP", "MISSING"):
            with self.subTest(status=status):
                coverage = inspection_evidence_coverage(
                    ({"tool_id": "db.storage.capacity", "check_status": status,
                      "evidence_refs": ["artifact:partial"]},),
                    (),
                )
                self.assertEqual(0.0, coverage["evidence_coverage_percent"])

    def test_unrelated_monitoring_gap_is_not_a_missing_database_check(self) -> None:
        coverage = inspection_evidence_coverage(
            ({"tool_id": "db.storage.capacity", "evidence_refs": ["artifact:capacity"]},),
            ({"step_id": "monitoring.coverage", "code": "MONITORING_WINDOW_INSUFFICIENT"},),
        )
        self.assertEqual(100.0, coverage["evidence_coverage_percent"])

    def test_zero_row_clear_finding_and_observation_can_all_be_complete(self) -> None:
        for status in ("CLEAR", "FINDING", "OBSERVED", "NOT_APPLICABLE"):
            with self.subTest(status=status):
                coverage = inspection_evidence_coverage(
                    ({"tool_id": "db.session.blocking_chain", "check_status": status,
                      "evidence_refs": ["artifact:zero-rows"]},),
                    (),
                )
                self.assertEqual(100.0, coverage["evidence_coverage_percent"])

    def test_empty_inspection_is_not_reported_as_full_coverage(self) -> None:
        coverage = inspection_evidence_coverage((), ())
        self.assertEqual(0, coverage["evidence_total_count"])
        self.assertEqual(0.0, coverage["evidence_coverage_percent"])
        self.assertEqual(0.0, coverage["tool_execution_coverage_percent"])


if __name__ == "__main__":
    unittest.main()
