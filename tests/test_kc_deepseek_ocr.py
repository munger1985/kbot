import unittest
from types import SimpleNamespace

from PIL import Image

from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.deepseek_ocr import (
    DeepSeekOcrClient,
    DeepSeekOcrElement,
    DeepSeekOcrEnrichmentResult,
    DeepSeekOcrPageResult,
    KcDeepSeekOcrEnricher,
)
from tests.test_kc_docling_atom_normalizer import FakeDocument, item


class FakeDeepSeekOcrClient:
    page_prompt = "<|grounding|>Convert the document to markdown."

    def __init__(self):
        self.prompts = []

    async def recognize(
        self,
        *,
        served_model_name,
        image,
        prompt,
    ):
        self.prompts.append(prompt)
        return (
            "title[[100, 100, 900, 180]]\n设备报告\n"
            "text[[100, 220, 900, 400]]\n实例状态正常"
        )


class DeepSeekOcrTest(unittest.IsolatedAsyncioTestCase):
    def test_parses_grounding_blocks_and_normalizes_bbox(self):
        elements = KcDeepSeekOcrEnricher.parse_grounding_elements(
            "title[[100, 50, 900, 120]]\n巡检报告\n"
            "table[[80, 180, 920, 800]]\n"
            "| 指标 | 数值 |\n| --- | --- |\n| CPU | 20% |"
        )

        self.assertEqual(
            [element.element_type for element in elements],
            ["title", "table"],
        )
        self.assertAlmostEqual(elements[0].bbox[0], 100 / 999)
        self.assertIn("CPU", elements[1].content_text)

    def test_stream_decoder_supports_delta_and_done(self):
        self.assertEqual(
            ["设备"],
            DeepSeekOcrClient._decode_stream_line(
                b'data: {"choices":[{"delta":{"content":"\\u8bbe\\u5907"}}]}'
            ),
        )
        self.assertEqual(
            [],
            DeepSeekOcrClient._decode_stream_line(b"data: [DONE]"),
        )

    async def test_enricher_selects_page_without_docling_text(self):
        page = SimpleNamespace(
            page_no=1,
            image=SimpleNamespace(
                pil_image=Image.new("RGB", (100, 100), "white")
            ),
        )
        document = SimpleNamespace(
            pages={1: page},
            pictures=[],
            iterate_items=lambda: iter(()),
        )
        client = FakeDeepSeekOcrClient()

        result = await KcDeepSeekOcrEnricher(
            client=client,
        ).enrich(
            document,
            served_model_name="deepseek-ocr-2",
        )

        self.assertEqual([page.page_no for page in result.page_results], [1])
        self.assertEqual(
            result.page_results[0].elements[0].element_type,
            "title",
        )
        self.assertIn("<|grounding|>", client.prompts[0])

    def test_pipeline_uses_ocr_atoms_with_grounding_locator(self):
        document = FakeDocument([
            item(
                source_ref="#/texts/0",
                label="text",
                text="错误文本",
            ),
        ])
        enrichment = DeepSeekOcrEnrichmentResult(
            served_model_name="deepseek-ocr-2",
            page_results=(
                DeepSeekOcrPageResult(
                    page_no=1,
                    elements=(
                        DeepSeekOcrElement(
                            element_type="title",
                            content_text="设备报告",
                            bbox=(0.1, 0.1, 0.9, 0.2),
                        ),
                        DeepSeekOcrElement(
                            element_type="text",
                            content_text="序列号 SN-001 运行正常",
                            bbox=(0.1, 0.25, 0.9, 0.4),
                        ),
                    ),
                    raw_output="grounding output",
                ),
            ),
            failed_page_numbers=(),
            picture_description_count=0,
        )

        output = KcParsingPipeline(parser_version="4.0-test").parse(
            document_version_id=10,
            parse_view_id=20,
            document=document,
            ocr_enrichment=enrichment,
        )

        contents = [
            evidence.content_text
            for evidence in output.evidences
        ]
        self.assertNotIn("错误文本", contents)
        self.assertTrue(any("SN-001" in content for content in contents))
        self.assertEqual(
            output.evidences[0].locator["pages"][0]["bbox"],
            [0.1, 0.1, 0.9, 0.2],
        )
        self.assertIn("deepseek_ocr_analysis", output.artifacts)
        self.assertEqual(
            "deepseek-ocr-2",
            output.quality_report.metrics["deepseek_ocr"]["model"],
        )


if __name__ == "__main__":
    unittest.main()
