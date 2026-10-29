"""智能诊断面向用户文本的展示字典测试。"""

import importlib
import unittest
from enum import Enum

from aiops_agent.application.display_dictionary import (
    AIOPS_DISPLAY_NAMES,
    AIOPS_METRIC_DISPLAY_NAMES,
    dictionary_for_payload,
    display_field_name,
    display_metric_name,
    display_text,
    format_metric_value,
)
from aiops_agent.application.inspections.check_catalog import load_check_catalog


class AIOpsDisplayDictionaryTest(unittest.TestCase):
    def test_answer_exposed_enums_have_display_names(self) -> None:
        module_names = (
            "platform_core.contracts.aiops.configuration",
            "platform_core.contracts.aiops.conversation",
            "platform_core.contracts.aiops.findings",
            "platform_core.contracts.aiops.investigation",
            "platform_core.contracts.aiops.playbooks",
            "platform_core.contracts.aiops.types",
            "platform_core.contracts.aiops.work_items",
            "platform_core.contracts.aiops.workload",
            "aiops_agent.contracts.implementation",
            "aiops_agent.domain.states",
            "aiops_agent.domain.evidence.events",
        )
        enum_types = {
            value
            for module_name in module_names
            for value in vars(importlib.import_module(module_name)).values()
            if isinstance(value, type)
            and issubclass(value, Enum)
            and value is not Enum
        }
        missing = {
            member.value
            for enum_type in enum_types
            for member in enum_type
            if member.value not in AIOPS_DISPLAY_NAMES
        }

        self.assertEqual(set(), missing)

    def test_registered_codes_are_fully_translated_in_answer_text(self) -> None:
        rendered = display_text(
            "当前结论为 PARTIAL，证据状态为 NEEDS_EVIDENCE，"
            "根因等级为 INCONCLUSIVE，缺口为 SOURCE_NO_DATA。"
        )

        self.assertEqual(
            "当前结论为 部分完成，证据状态为 仍需补充证据，"
            "根因等级为 证据不足，无法定论，缺口为 监控源未返回有效数据。",
            rendered,
        )
        for code in (
            "PARTIAL",
            "NEEDS_EVIDENCE",
            "INCONCLUSIVE",
            "SOURCE_NO_DATA",
        ):
            self.assertNotIn(code, rendered)

    def test_monitoring_evidence_codes_have_user_facing_labels(self) -> None:
        codes = (
            "MONITORING_UNAVAILABLE",
            "MONITORING_FORECAST_INSUFFICIENT",
            "MONITORING_FORECAST_LOW_CONFIDENCE",
            "MONITORING_COVERAGE_INSUFFICIENT",
            "MONITORING_WINDOW_INSUFFICIENT",
            "MONITORING_EVIDENCE_INSUFFICIENT",
        )

        rendered = display_text("、".join(codes))

        for code in codes:
            self.assertNotIn(code, rendered)

    def test_model_receives_only_relevant_dictionary_entries(self) -> None:
        payload = {
            "status": "PARTIAL",
            "finding": {"severity": "HIGH"},
            "sql_id": "abc123",
        }

        self.assertEqual(
            {"HIGH": "高", "PARTIAL": "部分完成"},
            dictionary_for_payload(payload),
        )

    def test_translation_preserves_inline_and_fenced_code(self) -> None:
        rendered = display_text(
            "状态为 ACTIVE，执行 `STATUS = 'ACTIVE'`：\n"
            "```sql\nWHERE STATUS = 'ACTIVE'\n```"
        )

        self.assertIn("状态为 启用", rendered)
        self.assertIn("`STATUS = 'ACTIVE'`", rendered)
        self.assertIn("WHERE STATUS = 'ACTIVE'", rendered)

    def test_metric_expression_is_rendered_for_formal_report(self) -> None:
        rendered = display_text(
            "db.storage.free_bytes[instance=oracle-aiops-88, job=oracle, "
            "tablespace=SYSTEM, type=PERMANENT]："
            "first=10，latest=20，change_per_day=2"
        )

        self.assertIn("存储可用容量", rendered)
        self.assertIn("表空间：SYSTEM", rendered)
        self.assertIn("表空间类型：永久表空间", rendered)
        self.assertIn("期初值：10", rendered)
        self.assertIn("当前值：20", rendered)
        self.assertIn("日均变化量：2", rendered)
        for internal_text in (
            "db.storage.free_bytes", "instance=", "job=", "type=",
            "first=", "latest=", "change_per_day=",
        ):
            self.assertNotIn(internal_text, rendered)

    def test_every_catalog_metric_has_a_human_readable_name(self) -> None:
        self.assertTrue(AIOPS_METRIC_DISPLAY_NAMES)
        for metric_code, expected in AIOPS_METRIC_DISPLAY_NAMES.items():
            self.assertEqual(expected, display_metric_name(metric_code))
            self.assertNotEqual(metric_code, expected)

    def test_unknown_metric_code_is_not_exposed(self) -> None:
        self.assertEqual("监控指标", display_metric_name("vendor.internal.metric"))

    def test_metric_bytes_are_rendered_in_readable_units(self) -> None:
        self.assertEqual("60.00 GiB", format_metric_value(60 * 1024 ** 3, "bytes"))

    def test_structured_field_assignment_is_not_exposed_as_program_variable(self) -> None:
        rendered = display_text("log_mode=NOARCHIVELOG，unknown_vendor_field=1")

        self.assertEqual("日志模式：NOARCHIVELOG，观测字段：1", rendered)
        self.assertNotIn("log_mode", rendered)
        self.assertNotIn("unknown_vendor_field", rendered)

    def test_plan_terms_and_object_names_are_not_translated_as_status_codes(self) -> None:
        rendered = display_text("FIXED TABLE FULL → SYS.TABLE → schema.TABLE；FIXED；TABLE；`TABLE`")
        self.assertEqual("FIXED TABLE FULL → SYS.TABLE → schema.TABLE；已修复；数据表格；`TABLE`", rendered)
        self.assertEqual("```sql\nSELECT FIXED FROM SYS.TABLE\n```", display_text("```sql\nSELECT FIXED FROM SYS.TABLE\n```"))

    def test_all_inspection_report_columns_have_human_readable_labels(self) -> None:
        columns = {
            column
            for group in load_check_catalog().groups
            for check in group.checks
            for column in check.report_columns
        }

        self.assertTrue(columns)
        self.assertEqual(
            set(),
            {column for column in columns if display_field_name(column) == "观测字段"},
        )


if __name__ == "__main__":
    unittest.main()
