"""智能诊断面向用户文本的展示字典测试。"""

import importlib
import unittest
from enum import Enum

from aiops_agent.application.display_dictionary import (
    AIOPS_DISPLAY_NAMES,
    dictionary_for_payload,
    display_text,
)
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


if __name__ == "__main__":
    unittest.main()
