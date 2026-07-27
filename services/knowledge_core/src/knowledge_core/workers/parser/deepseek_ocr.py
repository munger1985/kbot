"""Knowledge Core Parser 专用的 DeepSeek OCR 客户端与页面增强器。"""

import asyncio
from dataclasses import asdict, dataclass
import json
import re
from typing import Any

import aiohttp
from loguru import logger
from PIL import Image

from platform_core.codec.encoder import ImageEncoder


_GROUNDING_BLOCK = re.compile(
    r"^(text|title|table_caption|table|image|code|formula)"
    r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]",
    re.MULTILINE,
)
_INLINE_GROUNDING = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>",
)


@dataclass(frozen=True)
class DeepSeekOcrElement:
    element_type: str
    content_text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class DeepSeekOcrPageResult:
    page_no: int
    elements: tuple[DeepSeekOcrElement, ...]
    raw_output: str


@dataclass(frozen=True)
class DeepSeekOcrEnrichmentResult:
    served_model_name: str
    page_results: tuple[DeepSeekOcrPageResult, ...]
    failed_page_numbers: tuple[int, ...]
    picture_description_count: int
    adapter_version: str = "deepseek-ocr/v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeepSeekOcrClient:
    """直接调用独立 DeepSeek OCR OpenAI 兼容端点。"""

    def __init__(
        self,
        *,
        api_endpoint: str,
        timeout: int,
        crop_mode: bool,
        max_tokens: int,
        temperature: float,
    ):
        self._api_endpoint = api_endpoint
        self._timeout = timeout
        self._crop_mode = crop_mode
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def page_prompt(self) -> str:
        if self._crop_mode:
            return "<|grounding|>Convert the document to markdown."
        return "Convert the document to markdown."

    async def recognize(
        self,
        *,
        served_model_name: str,
        image: Image.Image,
        prompt: str,
    ) -> str:
        image_base64 = await ImageEncoder.encode(image)
        payload = {
            "model": served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/jpeg;base64,"
                                    f"{image_base64}"
                                )
                            },
                        },
                    ],
                }
            ],
            "stream": True,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self._api_endpoint,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                ) as response:
                    if response.status != 200:
                        detail = await response.text()
                        raise RuntimeError(
                            "DeepSeek OCR 服务返回 "
                            f"HTTP {response.status}：{detail[:500]}"
                        )
                    output: list[str] = []
                    while not response.content.at_eof():
                        raw_line = await response.content.readline()
                        if not raw_line:
                            break
                        output.extend(self._decode_stream_line(raw_line))
                    return "".join(output).strip()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"DeepSeek OCR 服务响应超时（{self._timeout} 秒）"
            ) from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError(
                f"无法连接 DeepSeek OCR 服务：{self._api_endpoint}"
            ) from exc

    @staticmethod
    def _decode_stream_line(raw_line: bytes | str) -> list[str]:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes)
            else raw_line
        ).strip()
        if not line or line == "data: [DONE]":
            return []
        if line.startswith("data: "):
            line = line[6:]
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        output: list[str] = []
        for choice in payload.get("choices", ()):
            content = (
                choice.get("delta", {}).get("content")
                or choice.get("message", {}).get("content")
                or ""
            )
            if content:
                output.append(str(content))
        return output


class KcDeepSeekOcrEnricher:
    """在 Docling 关闭内置 OCR 后补充页面和嵌入图片文字。"""

    def __init__(
        self,
        *,
        client: DeepSeekOcrClient,
        max_concurrency: int = 2,
        min_page_text_characters: int = 40,
    ):
        self._client = client
        self._max_concurrency = max_concurrency
        self._min_page_text_characters = min_page_text_characters

    async def enrich(
        self,
        document,
        *,
        served_model_name: str | None,
    ) -> DeepSeekOcrEnrichmentResult | None:
        if not served_model_name:
            return None
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

        async def recognize_page(
            page_no: int,
            page,
        ) -> DeepSeekOcrPageResult:
            async with semaphore:
                raw = await self._client.recognize(
                    served_model_name=served_model_name,
                    image=self._prepare_image(page.image.pil_image),
                    prompt=self._client.page_prompt,
                )
            elements = self.parse_grounding_elements(raw)
            if not elements:
                raise ValueError(f"第 {page_no} 页 OCR 结果为空")
            return DeepSeekOcrPageResult(
                page_no=page_no,
                elements=elements,
                raw_output=raw,
            )

        page_outputs = await asyncio.gather(
            *(
                recognize_page(page_no, page)
                for page_no, page in selected_pages
            ),
            return_exceptions=True,
        )
        page_results: list[DeepSeekOcrPageResult] = []
        failed_pages: list[int] = []
        for (page_no, _), output in zip(selected_pages, page_outputs):
            if isinstance(output, Exception):
                failed_pages.append(page_no)
                logger.warning(
                    "KC 第 {} 页 DeepSeek OCR 识别失败：{}",
                    page_no,
                    output,
                )
            else:
                page_results.append(output)

        selected_page_numbers = {
            page_no for page_no, _ in selected_pages
        }
        picture_results = await asyncio.gather(
            *(
                self._enrich_picture(
                    picture,
                    served_model_name=served_model_name,
                    selected_page_numbers=selected_page_numbers,
                    semaphore=semaphore,
                )
                for picture in getattr(document, "pictures", ())
            ),
            return_exceptions=True,
        )
        for output in picture_results:
            if isinstance(output, Exception):
                logger.warning("KC 图片 DeepSeek OCR 识别失败：{}", output)

        return DeepSeekOcrEnrichmentResult(
            served_model_name=served_model_name,
            page_results=tuple(page_results),
            failed_page_numbers=tuple(failed_pages),
            picture_description_count=sum(
                output is True for output in picture_results
            ),
        )

    async def _enrich_picture(
        self,
        picture,
        *,
        served_model_name: str,
        selected_page_numbers: set[int],
        semaphore: asyncio.Semaphore,
    ) -> bool:
        from docling_core.types.doc.document import DescriptionAnnotation

        image = getattr(getattr(picture, "image", None), "pil_image", None)
        if image is None or self._belongs_to_selected_page(
            picture,
            selected_page_numbers,
        ):
            return False
        existing = [
            annotation
            for annotation in getattr(picture, "annotations", ())
            if isinstance(annotation, DescriptionAnnotation)
            and "ocr" in str(annotation.provenance).lower()
        ]
        if existing:
            return False
        async with semaphore:
            content = await self._client.recognize(
                served_model_name=served_model_name,
                image=self._prepare_image(image),
                prompt="Parse the figure.",
            )
        if not content.strip():
            return False
        picture.annotations.append(
            DescriptionAnnotation(
                text=content.strip(),
                provenance=f"deepseek_ocr:{served_model_name}",
            )
        )
        return True

    @staticmethod
    def parse_grounding_elements(
        raw: str,
    ) -> tuple[DeepSeekOcrElement, ...]:
        matches = list(_GROUNDING_BLOCK.finditer(raw))
        if not matches:
            clean = KcDeepSeekOcrEnricher._clean_content(raw)
            if not clean:
                return ()
            return (
                DeepSeekOcrElement(
                    element_type="text",
                    content_text=clean,
                    bbox=(0.0, 0.0, 1.0, 1.0),
                ),
            )

        output: list[DeepSeekOcrElement] = []
        for index, match in enumerate(matches):
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(raw)
            )
            text = KcDeepSeekOcrEnricher._clean_content(
                raw[match.end():end]
            )
            if not text:
                continue
            coordinates = tuple(
                max(0.0, min(1.0, int(match.group(group)) / 999.0))
                for group in range(2, 6)
            )
            x0, y0, x1, y1 = coordinates
            output.append(
                DeepSeekOcrElement(
                    element_type=match.group(1),
                    content_text=text,
                    bbox=(
                        min(x0, x1),
                        min(y0, y1),
                        max(x0, x1),
                        max(y0, y1),
                    ),
                )
            )
        return tuple(output)

    @staticmethod
    def _clean_content(value: str) -> str:
        clean = _INLINE_GROUNDING.sub("", value)
        return re.sub(r"\n{3,}", "\n\n", clean).strip()

    @staticmethod
    def _text_characters_by_page(document) -> dict[int, int]:
        output: dict[int, int] = {}
        for item, _ in document.iterate_items():
            text = str(getattr(item, "text", "") or "").strip()
            if not text:
                continue
            for provenance in getattr(item, "prov", ()) or ():
                page_no = int(provenance.page_no)
                output[page_no] = output.get(page_no, 0) + len(text)
        return output

    @staticmethod
    def _prepare_image(image: Image.Image) -> Image.Image:
        prepared = image
        if max(prepared.size) > 1024:
            scale = 1024 / max(prepared.size)
            prepared = prepared.resize(
                (
                    max(1, int(prepared.width * scale)),
                    max(1, int(prepared.height * scale)),
                ),
                Image.Resampling.BICUBIC,
            )
        return prepared.convert("RGB") if prepared.mode != "RGB" else prepared

    @staticmethod
    def _belongs_to_selected_page(
        picture,
        selected_page_numbers: set[int],
    ) -> bool:
        return any(
            int(provenance.page_no) in selected_page_numbers
            for provenance in getattr(picture, "prov", ()) or ()
        )
