"""图片提取器 — 解析 VLM Markdown 中的 [IMAGE:xxx] 标记，保存页面截图。

用于 vlm 模式：VLM 在 Markdown 中插入 [IMAGE:描述] 标记，
此模块解析标记并保存对应页面的完整截图供后续 Phase 2 视觉检索使用。
"""

import os
import re
from dataclasses import dataclass

from loguru import logger
from docling_core.types.doc.document import DoclingDocument

from .precision_analyzer import PageMarkdown


IMAGE_MARKER_RE = re.compile(r'\[IMAGE:\s*(.+?)\]')


@dataclass
class ExtractedImage:
    """提取的图片记录"""
    page_no: int
    description: str
    image_path: str          # 本地文件路径


class ImageExtractor:
    """解析 [IMAGE:xxx] 标记，保存对应页面截图。

    用法:
        extractor = ImageExtractor()
        images = extractor.extract(doc=doc, pages=pages, file_id=file_id)
    """

    def extract(
        self,
        doc: DoclingDocument,
        pages: list[PageMarkdown],
        file_id: str,
        output_base: str = "",
    ) -> list[ExtractedImage]:
        """扫描 Markdown 中的 [IMAGE:xxx] 标记，保存对应页面截图。

        Args:
            doc: DoclingDocument（含 page images）
            pages: VLM 生成的 PageMarkdown 列表
            file_id: 文件 ID
            output_base: 输出根目录，默认为 knowledge_base/{file_id}

        Returns:
            提取的图片记录列表
        """
        if not output_base:
            output_base = f"knowledge_base/{file_id}"

        image_dir = os.path.join(output_base, "extracted_images")
        os.makedirs(image_dir, exist_ok=True)

        extracted: list[ExtractedImage] = []
        page_images = {p.page: p for p in pages}

        for p in pages:
            if not p.meaningful or not p.markdown:
                continue

            markers = IMAGE_MARKER_RE.findall(p.markdown)
            if not markers:
                continue

            page_obj = doc.pages.get(p.page)
            if not page_obj or not page_obj.image or not page_obj.image.pil_image:
                logger.warning(f"[ImageExtractor] 第 {p.page} 页无页面图，跳过 [IMAGE] 标记")
                continue

            for idx, desc in enumerate(markers):
                desc = desc.strip()
                img_filename = f"page{p.page}_img{idx+1}.png"
                img_path = os.path.join(image_dir, img_filename)

                try:
                    page_obj.image.pil_image.save(img_path)
                    extracted.append(ExtractedImage(
                        page_no=p.page,
                        description=desc,
                        image_path=img_path,
                    ))
                    logger.debug(
                        f"[ImageExtractor] [IMAGE:{desc}] → {img_path}"
                    )
                except Exception as e:
                    logger.error(f"[ImageExtractor] 保存图片失败: {e}")

        if extracted:
            logger.success(
                f"[ImageExtractor] 提取 {len(extracted)} 张图片 "
                f"→ {image_dir}"
            )

        return extracted
