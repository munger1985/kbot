"""仅供 KC Parser Worker 使用的 Docling 转换适配器。"""

import asyncio
from pathlib import Path
import subprocess
import tempfile

# 先加载 Conda 的 Tesseract 图像库，避免 Docling 后续加载的 torchvision
# 内置 libjpeg.so.8 抢占同名动态库，导致 libtiff 缺少 jpeg12 符号。
try:
    import tesserocr as _tesserocr_runtime
except ModuleNotFoundError:
    _tesserocr_runtime = None

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractOcrOptions,
)
from docling.document_converter import (
    CsvFormatOption,
    DocumentConverter,
    ExcelFormatOption,
    HTMLFormatOption,
    ImageFormatOption,
    MarkdownFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling_core.types.doc import DoclingDocument


class KcDoclingConverter:
    def __init__(self, *, artifacts_path: str):
        resolved_path = Path(artifacts_path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(
                f"Docling 模型根目录不存在：{resolved_path}"
            )
        self._artifacts_path = resolved_path

    async def convert(
        self, *, source_path: Path, do_ocr: bool = True,
        ocr_engine: str = "tesseract", image_scale: float = 2.0,
        generate_page_images: bool = False,
        generate_picture_images: bool = True,
        generate_table_images: bool = False,
        detect_table_structure: bool = True,
    ) -> DoclingDocument:
        return await asyncio.to_thread(
            self._convert_sync, source_path, do_ocr, ocr_engine, image_scale,
            generate_page_images, generate_picture_images,
            generate_table_images, detect_table_structure,
        )

    def _convert_sync(
        self, source_path: Path, do_ocr: bool, ocr_engine: str,
        image_scale: float, generate_page_images: bool,
        generate_picture_images: bool, generate_table_images: bool,
        detect_table_structure: bool,
    ) -> DoclingDocument:
        table_options = TableStructureOptions(do_cell_matching=True)
        pipeline_options = PdfPipelineOptions(
            artifacts_path=self._artifacts_path,
            do_ocr=do_ocr,
            do_chart_extraction=False,
            generate_table_images=generate_table_images,
            generate_picture_images=generate_picture_images,
            generate_page_images=generate_page_images,
            table_structure_options=table_options,
            images_scale=image_scale,
        )
        pipeline_options.do_table_structure = detect_table_structure
        if ocr_engine.lower() == "tesseract":
            ocr_options = TesseractOcrOptions(lang=["chi_sim", "eng"])
        else:
            ocr_options = EasyOcrOptions(lang=["ch_sim", "en"])
        ocr_options.force_full_page_ocr = False
        pipeline_options.ocr_options = ocr_options
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=pipeline_options),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=pipeline_options),
            InputFormat.XLSX: ExcelFormatOption(pipeline_options=pipeline_options),
            InputFormat.MD: MarkdownFormatOption(),
            InputFormat.HTML: HTMLFormatOption(),
            InputFormat.CSV: CsvFormatOption(),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        })
        suffix = source_path.suffix.lower()
        if suffix not in {".ppt", ".pptx", ".doc", ".xls"}:
            return converter.convert(source_path).document
        with tempfile.TemporaryDirectory(prefix="kbot-kc-ppt-") as temporary_directory:
            profile_uri = (Path(temporary_directory) / "libreoffice-profile").as_uri()
            target_format = {".ppt": "pdf", ".pptx": "pdf", ".doc": "docx", ".xls": "xlsx"}[suffix]
            subprocess.run([
                "soffice", f"-env:UserInstallation={profile_uri}",
                "--headless", "--convert-to", target_format,
                "--outdir", temporary_directory, str(source_path),
            ], check=True, capture_output=True, timeout=120)
            converted_path = Path(temporary_directory) / f"{source_path.stem}.{target_format}"
            if not converted_path.is_file():
                raise RuntimeError(f"LibreOffice did not produce the expected {target_format} file")
            return converter.convert(converted_path).document
