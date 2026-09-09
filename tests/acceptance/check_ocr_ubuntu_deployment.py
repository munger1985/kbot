"""检查 Ubuntu OCR 安装器包含 Model Serving 与 Docling 共同所需的依赖。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts/deployment/install_model_ocr_ubuntu24.sh"
UNIT_TEMPLATE = ROOT / "scripts/deployment/systemd/kbot-model-ocr.service.template"


class OcrUbuntuDeploymentTests(unittest.TestCase):
    """阻止 OCR 独立进程与 Docling 采用不一致的 Tesseract 运行时。"""

    def test_installer_provisions_shared_ocr_dependencies(self) -> None:
        source = INSTALLER.read_text(encoding="utf-8")

        self.assertIn("tesseract tesserocr", source)
        self.assertIn("libtiff libjpeg-turbo", source)
        self.assertIn("--strict-channel-priority --force-reinstall", source)
        self.assertIn("--override-channels -c conda-forge", source)
        self.assertNotIn("'onnxruntime=1.29.0=*_cpu'", source)
        self.assertIn("--force-reinstall --no-deps onnxruntime==1.29.0", source)
        self.assertIn("scripts/deployment/install_workspace.sh", source)
        self.assertIn("--skip-runtime-install", source)
        self.assertIn("--download-models", source)
        self.assertIn("docling-tools models download rapidocr easyocr", source)
        self.assertIn('${KBOT_DOCLING_MODELS_DIR:-$HOME/models/docling_models}', source)
        self.assertIn('"$HOME/miniconda3/bin/conda"', source)
        self.assertIn('"/opt/miniconda3/bin/conda"', source)
        self.assertIn("RapidOCR 本地模型加载验证通过", source)
        self.assertIn("EasyOCR 本地模型加载验证通过", source)
        self.assertIn("Tesseract Python 绑定导入通过", source)
        self.assertNotIn("import easyocr\nimport onnxruntime\nimport tesserocr", source)

    def test_systemd_unit_uses_the_same_tessdata_environment(self) -> None:
        source = UNIT_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("TESSDATA_PREFIX=@TESSDATA_PREFIX@", source)
        self.assertIn("model_serving.entrypoints.ocr", source)
        self.assertIn("EnvironmentFile=@KBOT_SOURCE_ROOT@/.env", source)


if __name__ == "__main__":
    unittest.main()
