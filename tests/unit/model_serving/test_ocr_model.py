"""OCR Provider 的独立模型池和本地文字识别归一化测试。"""

from __future__ import annotations

from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from model_serving.common.provider_catalog import (
    list_provider_options,
    validate_provider_config,
)
from model_serving.ocr.model import OCRModel
from platform_core.dictionary import ModelCategory, OCRProvider


def _model(provider: str, params: dict | None = None) -> OCRModel:
    return OCRModel(model_data={
        "model_id": "018f3de2-01d5-7e89-8c41-1c7af92b3a11",
        "provider": provider,
        "provider_model_name": "ocr-test",
        "model_params": params or {},
    })


class OCRModelTest(unittest.TestCase):
    def test_tesseract_tsv_is_normalized_to_text_and_blocks(self):
        model = _model(
            OCRProvider.LOCAL_TESSERACT.value,
            {"languages": ["chi_sim", "eng"], "model_path": "/models/tessdata"},
        )
        model._engine = "tesseract"
        output = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t40\t96.2\tORA-01017\n"
            "5\t1\t1\t1\t1\t2\t42\t20\t18\t40\t91.0\tinvalid\n"
        ).encode()
        with patch("model_serving.ocr.model.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout=output)
            text, blocks = model._run_tesseract(b"image")
        self.assertEqual("ORA-01017 invalid", text)
        self.assertEqual("ORA-01017", blocks[0]["text"])
        self.assertEqual(10, blocks[0]["bbox"]["left"])
        self.assertEqual(["tesseract", "stdin", "stdout", "-l", "chi_sim+eng", "--tessdata-dir", "/models/tessdata", "tsv"], run.call_args.args[0])

    def test_tesseract_failure_does_not_expose_engine_stderr(self):
        model = _model(OCRProvider.LOCAL_TESSERACT.value)
        model._engine = "tesseract"
        with patch(
            "model_serving.ocr.model.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["tesseract"], stderr=b"sensitive"),
        ):
            with self.assertRaisesRegex(RuntimeError, "推理失败") as raised:
                model._run_tesseract(b"image")
        self.assertNotIn("sensitive", str(raised.exception))

    def test_easyocr_result_preserves_polygon_and_confidence(self):
        text, blocks = OCRModel._normalize_easyocr([
            ([[1, 2], [3, 4], [5, 6], [7, 8]], "数据库", 0.88),
        ])
        self.assertEqual("数据库", text)
        self.assertEqual(0.88, blocks[0]["confidence"])
        self.assertEqual([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], blocks[0]["polygon"])

    def test_deepseek_stream_and_grounding_are_decoded(self):
        model = _model(OCRProvider.API_DEEPSEEK_OCR.value)
        self.assertEqual(["标题"], model._decode_deepseek_stream_line(
            'data: {"choices":[{"delta":{"content":"标题"}}]}'.encode()
        ))
        blocks = model._parse_deepseek_grounding_blocks(
            "text[[1, 2, 3, 4]]标题"
        )
        self.assertEqual([1, 2, 3, 4], blocks[0]["bbox"])
        self.assertEqual("标题", blocks[0]["text"])


class OCRProviderCatalogTest(unittest.TestCase):
    def test_catalog_lists_four_ocr_providers(self):
        providers = {
            item.provider
            for item in list_provider_options(category=ModelCategory.OCR.value)
        }
        self.assertEqual({
            "local_rapidocr", "local_easyocr", "local_tesseract",
            "api_deepseek_ocr",
        }, providers)

    def test_local_easyocr_requires_model_path(self):
        with self.assertRaisesRegex(ValueError, "model_path"):
            validate_provider_config({
                "category": ModelCategory.OCR.value,
                "provider": "local_easyocr",
                "model_params": {},
            })


if __name__ == "__main__":
    unittest.main()
