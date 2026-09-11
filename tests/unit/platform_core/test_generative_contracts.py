"""验证扩展生成能力的供应商中立内部契约。"""

from datetime import date
import unittest

from pydantic import ValidationError

from platform_core.contracts import ImageGenerationRequest, ResearchRequest
from platform_core.identity import uuid7


class GenerativeContractTests(unittest.TestCase):
    """防止业务 App 透传不受控的供应商 JSON。"""

    def test_research_rejects_mutually_exclusive_x_filters(self):
        with self.assertRaises(ValidationError):
            ResearchRequest(
                model_id=uuid7(), input="查找最近动态", trace_id="trace",
                allowed_x_handles=("xai",), excluded_x_handles=("grok",),
            )

    def test_research_rejects_invalid_date_range_and_too_many_handles(self):
        with self.assertRaises(ValidationError):
            ResearchRequest(
                model_id=uuid7(), input="查找最近动态", trace_id="trace",
                from_date=date(2026, 9, 2), to_date=date(2026, 9, 1),
            )
        with self.assertRaises(ValidationError):
            ResearchRequest(
                model_id=uuid7(), input="查找最近动态", trace_id="trace",
                allowed_x_handles=tuple(f"user{index}" for index in range(21)),
            )

    def test_image_request_has_bounded_supplier_neutral_inputs(self):
        request = ImageGenerationRequest(
            model_id=uuid7(), prompt="水墨风格的智能工作台", trace_id="trace",
        )

        self.assertEqual("1:1", request.aspect_ratio)
        self.assertEqual(1, request.count)
        with self.assertRaises(ValidationError):
            ImageGenerationRequest(
                model_id=uuid7(), prompt="图片", trace_id="trace", count=5,
            )
        with self.assertRaises(ValidationError):
            ImageGenerationRequest(
                model_id=uuid7(), prompt="图片", trace_id="trace", provider_tools={},
            )


if __name__ == "__main__":
    unittest.main()
