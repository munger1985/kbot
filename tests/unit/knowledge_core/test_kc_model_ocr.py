import unittest
from uuid import uuid4

from PIL import Image

from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.model_ocr import (
    KcModelOcrEnricher,
    OcrElement,
    OcrEnrichmentResult,
    OcrPageResult,
)
from test_kc_docling_atom_normalizer import FakeDocument, item


class ModelOcrTest(unittest.TestCase):
    def test_normalizes_model_serving_blocks(self):
        elements = KcModelOcrEnricher._elements(
            {
                "blocks": [
                    {"type": "title", "text": "巡检报告", "bbox": [100, 50, 900, 120]},
                    {"type": "table", "text": "| CPU | 20% |", "bbox": [80, 180, 920, 800]},
                ]
            },
            image=Image.new("RGB", (1000, 1000), "white"),
        )

        self.assertEqual([element.element_type for element in elements], ["title", "table"])
        self.assertAlmostEqual(elements[0].bbox[0], 100 / 999)
        self.assertIn("CPU", elements[1].content_text)

    def test_pipeline_uses_generic_ocr_atoms_and_artifact(self):
        document = FakeDocument([item(source_ref="#/texts/0", label="text", text="错误文本")])
        model_id = str(uuid4())
        enrichment = OcrEnrichmentResult(
            model_id=model_id,
            model_revision="revision-7",
            page_results=(OcrPageResult(
                page_no=1,
                elements=(
                    OcrElement("title", "设备报告", (0.1, 0.1, 0.9, 0.2)),
                    OcrElement("text", "序列号 SN-001 运行正常", (0.1, 0.25, 0.9, 0.4)),
                ),
                raw_output="ocr output",
            ),),
            failed_page_numbers=(),
            picture_description_count=0,
        )

        output = KcParsingPipeline(parser_version="1.0-test").parse(
            document_version_id=10,
            parse_view_id=20,
            document=document,
            ocr_enrichment=enrichment,
        )

        contents = [evidence.content_text for evidence in output.evidences]
        self.assertNotIn("错误文本", contents)
        self.assertTrue(any("SN-001" in content for content in contents))
        self.assertIn("ocr_analysis", output.artifacts)
        self.assertEqual(model_id, output.quality_report.metrics["ocr"]["model_id"])
        self.assertEqual("revision-7", output.quality_report.metrics["ocr"]["model_revision"])


if __name__ == "__main__":
    unittest.main()
