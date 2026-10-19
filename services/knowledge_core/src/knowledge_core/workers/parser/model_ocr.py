"""由 Collection 模型绑定驱动的 Knowledge Core OCR 增强。"""

import asyncio
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any
from uuid import UUID

from loguru import logger
from PIL import Image

from platform_clients import OCRClient
from platform_core.contracts import AuthContext, PrincipalKind


@dataclass(frozen=True)
class OcrElement:
    element_type: str
    content_text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OcrPageResult:
    page_no: int
    elements: tuple[OcrElement, ...]
    raw_output: str


@dataclass(frozen=True)
class OcrEnrichmentResult:
    model_id: str
    model_revision: str
    page_results: tuple[OcrPageResult, ...]
    failed_page_numbers: tuple[int, ...]
    picture_description_count: int
    adapter_version: str = "model-ocr/v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class KcModelOcrEnricher:
    """仅处理低文本页面和独立图片，模型身份来自 KC 任务快照。"""

    def __init__(
        self, *, client: OCRClient, caller_service: str,
        max_concurrency: int = 2, min_page_text_characters: int = 40,
    ):
        self._client = client
        self._caller_service = caller_service
        self._max_concurrency = max_concurrency
        self._min_page_text_characters = min_page_text_characters

    async def enrich(
        self, document, *, model_id: UUID | None,
    ) -> OcrEnrichmentResult | None:
        if model_id is None:
            return None
        context = AuthContext(
            principal_kind=PrincipalKind.SERVICE,
            client_id=self._caller_service,
            calling_service=self._caller_service,
            request_id=f"kc-ocr:{model_id}",
            trace_id=f"kc-ocr:{model_id}",
        )
        semaphore = asyncio.Semaphore(self._max_concurrency)
        text_by_page = self._text_characters_by_page(document)
        selected_pages = [
            (int(getattr(page, "page_no", key)), page)
            for key, page in sorted(document.pages.items())
            if (
                text_by_page.get(int(getattr(page, "page_no", key)), 0)
                < self._min_page_text_characters
                and getattr(getattr(page, "image", None), "pil_image", None)
                is not None
            )
        ]
        revisions: set[str] = set()

        async def recognize_page(page_no: int, page) -> OcrPageResult:
            image = self._prepare_image(page.image.pil_image)
            async with semaphore:
                result = await self._recognize(
                    model_id=model_id,
                    image=image,
                    context=context,
                )
            revisions.add(str(result.get("model_revision") or model_id))
            elements = self._elements(result, image=image)
            if not elements:
                raise ValueError(f"第 {page_no} 页 OCR 结果为空")
            return OcrPageResult(
                page_no=page_no,
                elements=elements,
                raw_output=str(result.get("text") or ""),
            )

        outputs = await asyncio.gather(
            *(recognize_page(page_no, page) for page_no, page in selected_pages),
            return_exceptions=True,
        )
        page_results: list[OcrPageResult] = []
        failed_pages: list[int] = []
        for (page_no, _), output in zip(selected_pages, outputs):
            if isinstance(output, Exception):
                failed_pages.append(page_no)
                logger.warning("KC 第 {} 页 OCR 识别失败：{}", page_no, output)
            else:
                page_results.append(output)
        selected_numbers = {page_no for page_no, _ in selected_pages}
        picture_outputs = await asyncio.gather(
            *(
                self._enrich_picture(
                    picture, model_id=model_id, context=context,
                    selected_page_numbers=selected_numbers,
                    semaphore=semaphore,
                )
                for picture in getattr(document, "pictures", ())
            ),
            return_exceptions=True,
        )
        for output in picture_outputs:
            if isinstance(output, Exception):
                logger.warning("KC 图片 OCR 识别失败：{}", output)
        return OcrEnrichmentResult(
            model_id=str(model_id),
            model_revision=sorted(revisions)[-1] if revisions else str(model_id),
            page_results=tuple(page_results),
            failed_page_numbers=tuple(failed_pages),
            picture_description_count=sum(output is True for output in picture_outputs),
        )

    async def _recognize(self, *, model_id: UUID, image: Image.Image, context):
        stream = BytesIO()
        image.save(stream, format="JPEG", quality=92)
        return await self._client.infer(
            model_id=model_id,
            image=stream.getvalue(),
            mime_type="image/jpeg",
            auth_context=context,
        )

    async def _enrich_picture(
        self, picture, *, model_id: UUID, context,
        selected_page_numbers: set[int], semaphore: asyncio.Semaphore,
    ) -> bool:
        from docling_core.types.doc.document import DescriptionAnnotation

        image = getattr(getattr(picture, "image", None), "pil_image", None)
        if image is None or self._belongs_to_selected_page(
            picture, selected_page_numbers,
        ):
            return False
        prepared = self._prepare_image(image)
        async with semaphore:
            result = await self._recognize(
                model_id=model_id, image=prepared, context=context,
            )
        text = str(result.get("text") or "").strip()
        if not text:
            return False
        picture.annotations.append(DescriptionAnnotation(
            text=text,
            provenance=f"model_ocr:{model_id}",
        ))
        return True

    @classmethod
    def _elements(
        cls, result: dict[str, Any], *, image: Image.Image,
    ) -> tuple[OcrElement, ...]:
        elements: list[OcrElement] = []
        supported_types = {
            "text", "title", "table_caption", "table", "image", "code", "formula",
        }
        for block in result.get("blocks") or ():
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "").strip()
            if not text:
                continue
            block_type = str(block.get("type") or "text")
            elements.append(OcrElement(
                element_type=block_type if block_type in supported_types else "text",
                content_text=text,
                bbox=cls._normalize_bbox(block, image=image),
            ))
        if elements:
            return tuple(elements)
        text = str(result.get("text") or "").strip()
        return (
            OcrElement("text", text, (0.0, 0.0, 1.0, 1.0)),
        ) if text else ()

    @staticmethod
    def _normalize_bbox(
        block: dict[str, Any], *, image: Image.Image,
    ) -> tuple[float, float, float, float]:
        width, height = max(1, image.width), max(1, image.height)
        bbox = block.get("bbox")
        if isinstance(bbox, dict):
            x0 = float(bbox.get("left") or 0)
            y0 = float(bbox.get("top") or 0)
            x1 = x0 + float(bbox.get("width") or 0)
            y1 = y0 + float(bbox.get("height") or 0)
            values = (x0 / width, y0 / height, x1 / width, y1 / height)
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            raw = tuple(float(value) for value in bbox)
            if max(raw) <= 1:
                values = raw
            elif max(raw) <= 1000:
                values = tuple(value / 999.0 for value in raw)
            else:
                values = (raw[0] / width, raw[1] / height, raw[2] / width, raw[3] / height)
        else:
            polygon = block.get("polygon")
            if not isinstance(polygon, (list, tuple)) or not polygon:
                return (0.0, 0.0, 1.0, 1.0)
            points = [
                point for point in polygon
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if not points:
                return (0.0, 0.0, 1.0, 1.0)
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            values = (
                min(xs) / width, min(ys) / height,
                max(xs) / width, max(ys) / height,
            )
        x0, y0, x1, y1 = (max(0.0, min(1.0, value)) for value in values)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        return image.convert("RGB")

    @staticmethod
    def _text_characters_by_page(document) -> dict[int, int]:
        counts: dict[int, int] = {}
        for item, _ in document.iterate_items():
            text = str(getattr(item, "text", "") or "").strip()
            for provenance in getattr(item, "prov", ()):
                page_no = int(getattr(provenance, "page_no", 0) or 0)
                if page_no > 0:
                    counts[page_no] = counts.get(page_no, 0) + len(text)
        return counts

    @staticmethod
    def _belongs_to_selected_page(picture, selected: set[int]) -> bool:
        return any(
            int(getattr(provenance, "page_no", 0) or 0) in selected
            for provenance in getattr(picture, "prov", ())
        )
