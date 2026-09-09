"""图片输入解析入口审计日志回归测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiops_agent.application.investigation.service import TurnPlanningService
from platform_core.identity import uuid7


class ImageInputGateLoggingTests(unittest.TestCase):
    def test_records_sanitized_gate_decision_for_unreferenced_image(self) -> None:
        service = TurnPlanningService.__new__(TurnPlanningService)
        service._conversation_input_resolver = object()
        context = SimpleNamespace(
            turn_id=uuid7(),
            ops_run_id=uuid7(),
            agent_id=uuid7(),
            trace_id="trace-image-gate",
            content=(
                {"content_type": "TEXT", "text": "分析截图"},
                {"content_type": "IMAGE", "media_type": "image/png"},
            ),
            image_capabilities={"vlm": {"default_model_id": str(uuid7())}},
        )

        with patch("aiops_agent.application.investigation.service.logger") as logger:
            has_upload_reference = service._log_image_input_resolution_gate(context)

        self.assertFalse(has_upload_reference)
        logger.info.assert_called_once()
        template, *values = logger.info.call_args.args
        self.assertIn("图片输入解析入口判定", template)
        self.assertIn("resolver_bound={}", template)
        self.assertIn("image_item_count={}", template)
        self.assertIn("upload_reference_count={}", template)
        self.assertIn("image_capability_keys={}", template)
        self.assertEqual(True, values[4])
        self.assertEqual(2, values[5])
        self.assertEqual(1, values[6])
        self.assertEqual(0, values[7])
        self.assertEqual(0, values[8])
        self.assertEqual("vlm", values[9])
        self.assertFalse(values[10])


if __name__ == "__main__":
    unittest.main()
