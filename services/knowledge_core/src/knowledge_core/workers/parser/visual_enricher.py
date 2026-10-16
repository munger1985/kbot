"""策略控制的局部图片与整页视觉增强。"""

import asyncio
from dataclasses import asdict, dataclass
import re
from typing import Any

from docling_core.types.doc.document import DescriptionAnnotation
from loguru import logger

from platform_clients import AIModelClient
from knowledge_core.workers.parser.visual_cache import LocalVisualResultCache


@dataclass(frozen=True)
class PageQualityAssessment:
    page_no: int
    text_characters: int
    mean_confidence: float | None
    gibberish_ratio: float
    image_available: bool
    requires_visual: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class VisualPageResult:
    page_no: int
    markdown: str
    served_model_name: str
    confidence: float
    replace_docling: bool
    selection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class VisualEnrichmentResult:
    strategy: str
    picture_description_count: int
    page_assessments: tuple[PageQualityAssessment, ...]
    page_results: tuple[VisualPageResult, ...]
    failed_page_numbers: tuple[int, ...]
    enabled: bool = True
    skip_reason: str | None = None
    adapter_version: str = "visual-fusion/v1"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


FULL_PAGE_MARKDOWN_PROMPT = """你是文档视觉解析器。请把整张页面转换为结构化 Markdown。

要求：
1. 原文照录文字、数字、单位和符号，不翻译、不概括；
2. 使用 #、##、### 表示真实标题层级；
3. 表格使用 Markdown 表格，不能省略行列；
4. 流程图、架构图和照片使用 > 描述其可见内容与关系；
5. 列表和公式保留原有结构；
6. 不编造页面中不存在的内容；
7. 直接输出 Markdown，不要输出代码块或额外解释。"""


class KcVisualEnricher:
    def __init__(self, client_factory=AIModelClient, cache=None):
        self._client_factory = client_factory
        self._cache = cache

    async def enrich(
        self,
        document,
        *,
        served_model_name: str | None,
        prompt: str,
        full_page_prompt: str = FULL_PAGE_MARKDOWN_PROMPT,
        strategy: str = "AUTO",
        min_text_characters: int = 80,
        min_mean_confidence: float = 0.65,
        max_gibberish_ratio: float = 0.08,
        max_concurrency: int = 2,
    ) -> VisualEnrichmentResult:
        normalized_strategy = strategy.strip().upper()
        if normalized_strategy not in {"TEXT", "AUTO", "VISUAL"}:
            raise ValueError(f"不支持的解析策略：{strategy}")
        assessments = self._assess_pages(
            document,
            min_text_characters=min_text_characters,
            min_mean_confidence=min_mean_confidence,
            max_gibberish_ratio=max_gibberish_ratio,
        )
        if not served_model_name:
            if normalized_strategy == "VISUAL":
                raise ValueError(
                    f"{normalized_strategy} 解析策略必须配置 VLM 模型"
                )
            return VisualEnrichmentResult(
                strategy=normalized_strategy,
                picture_description_count=0,
                page_assessments=assessments,
                page_results=(),
                failed_page_numbers=(),
                enabled=False,
                skip_reason="MODEL_NOT_CONFIGURED",
            )
        if normalized_strategy == "TEXT":
            return VisualEnrichmentResult(
                strategy=normalized_strategy,
                picture_description_count=0,
                page_assessments=assessments,
                page_results=(),
                failed_page_numbers=(),
                enabled=False,
                skip_reason="PARSE_STRATEGY_TEXT",
            )
        client = self._client_factory()
        full_page_prompt = full_page_prompt.strip() or FULL_PAGE_MARKDOWN_PROMPT
        semaphore = asyncio.Semaphore(max_concurrency)
        picture_memo: dict[str, str] = {}
        picture_locks: dict[str, asyncio.Lock] = {}

        async def enrich_picture(picture) -> bool:
            existing = [
                annotation for annotation in getattr(picture, "annotations", ())
                if isinstance(annotation, DescriptionAnnotation)
                and annotation.text and any(
                    marker in str(annotation.provenance).lower() for marker in ("vlm", "visual")
                )
            ]
            image = getattr(getattr(picture, "image", None), "pil_image", None)
            if existing or image is None or not self._is_informative_picture(image):
                return False
            options: dict[str, Any] = {}
            try:
                image_key = LocalVisualResultCache.image_sha256(image)
            except Exception:
                image_key = str(id(image))
            lock = picture_locks.setdefault(image_key, asyncio.Lock())
            async with lock:
                description = picture_memo.get(image_key)
                if description is None:
                    description = await self._cached_result(
                        image=image,
                        kind="picture",
                        model=served_model_name,
                        prompt=prompt,
                        options=options,
                    )
                if description is None:
                    async with semaphore:
                        description = await client.get_vlm_answer(
                            served_model_name, image, prompt=prompt,
                        )
                    await self._cache_result(
                        image=image,
                        kind="picture",
                        model=served_model_name,
                        prompt=prompt,
                        options=options,
                        result=description,
                    )
                picture_memo[image_key] = description
            if not description.strip():
                return False
            picture.annotations.append(DescriptionAnnotation(
                text=description.strip(),
                provenance=f"vlm_kc_v2:{served_model_name}",
            ))
            return True

        picture_results = (
            await asyncio.gather(
                *(
                    enrich_picture(picture)
                    for picture in getattr(document, "pictures", ())
                ),
                return_exceptions=True,
            )
            if normalized_strategy == "AUTO"
            else []
        )
        for failure in (
            result for result in picture_results
            if isinstance(result, Exception)
        ):
            logger.warning("KC 视觉描述生成失败：{}", failure)

        assessment_by_page = {
            assessment.page_no: assessment for assessment in assessments
        }
        if normalized_strategy == "VISUAL":
            selected = [
                assessment
                for assessment in assessments
                if assessment.image_available
            ]
        else:
            # AUTO is strictly object-level: never render or submit a full page.
            selected = []

        async def analyze_page(
            assessment: PageQualityAssessment,
        ) -> VisualPageResult:
            page = next(
                page
                for key, page in document.pages.items()
                if int(getattr(page, "page_no", key))
                == assessment.page_no
            )
            image = page.image.pil_image
            options = {"temperature": 0.1, "max_tokens": 4096}
            markdown = await self._cached_result(
                image=image,
                kind="page",
                model=served_model_name,
                prompt=full_page_prompt,
                options=options,
            )
            if markdown is None:
                async with semaphore:
                    markdown = await client.get_vlm_answer(
                        served_model_name,
                        image,
                        prompt=full_page_prompt,
                        **options,
                    )
                await self._cache_result(
                    image=image,
                    kind="page",
                    model=served_model_name,
                    prompt=full_page_prompt,
                    options=options,
                    result=markdown,
                )
            if not markdown.strip():
                raise ValueError(
                    f"第 {assessment.page_no} 页视觉解析结果为空"
                )
            return VisualPageResult(
                page_no=assessment.page_no,
                markdown=markdown.strip(),
                served_model_name=served_model_name,
                confidence=0.82,
                replace_docling=False,
                selection_reasons=assessment.reasons,
            )

        page_outputs = await asyncio.gather(
            *(analyze_page(assessment) for assessment in selected),
            return_exceptions=True,
        )
        page_results: list[VisualPageResult] = []
        failed_pages: list[int] = []
        for assessment, output in zip(selected, page_outputs):
            if isinstance(output, Exception):
                failed_pages.append(assessment.page_no)
                logger.warning(
                    "KC 第 {} 页整页视觉解析失败：{}",
                    assessment.page_no,
                    output,
                )
            else:
                page_results.append(output)
        return VisualEnrichmentResult(
            strategy=normalized_strategy,
            picture_description_count=sum(
                result is True for result in picture_results
            ),
            page_assessments=tuple(
                assessment_by_page[page_no]
                for page_no in sorted(assessment_by_page)
            ),
            page_results=tuple(
                sorted(page_results, key=lambda value: value.page_no)
            ),
            failed_page_numbers=tuple(failed_pages),
        )

    async def _cached_result(
        self, *, image, kind: str, model: str, prompt: str,
        options: dict[str, Any],
    ) -> str | None:
        if self._cache is None:
            return None
        try:
            return await self._cache.get(
                image=image, kind=kind, model=model, prompt=prompt,
                options=options,
            )
        except Exception as exc:
            logger.warning("KC 视觉缓存读取失败，继续调用模型：{}", exc)
            return None

    async def _cache_result(
        self, *, image, kind: str, model: str, prompt: str,
        options: dict[str, Any], result: str,
    ) -> None:
        if self._cache is not None and result.strip():
            try:
                await self._cache.put(
                    image=image, kind=kind, model=model, prompt=prompt,
                    options=options, result=result,
                )
            except Exception as exc:
                logger.warning("KC 视觉缓存写入失败，保留模型结果：{}", exc)

    @staticmethod
    def _is_informative_picture(image) -> bool:
        """Skip obvious icons and separator/header graphics before VLM."""
        size = getattr(image, "size", None)
        if not size or len(size) != 2:
            return True
        width, height = (int(size[0]), int(size[1]))
        if width < 48 or height < 48 or width * height < 16_384:
            return False
        return max(width / height, height / width) <= 12

    @classmethod
    def _assess_pages(
        cls,
        document,
        *,
        min_text_characters: int,
        min_mean_confidence: float,
        max_gibberish_ratio: float,
    ) -> tuple[PageQualityAssessment, ...]:
        text_by_page: dict[int, list[str]] = {}
        confidence_by_page: dict[int, list[float]] = {}
        for item, _ in document.iterate_items():
            text = str(getattr(item, "text", "") or "").strip()
            for provenance in getattr(item, "prov", ()) or ():
                page_no = int(provenance.page_no)
                if text:
                    text_by_page.setdefault(page_no, []).append(text)
                confidence = getattr(provenance, "confidence", None)
                if confidence is not None:
                    confidence_by_page.setdefault(page_no, []).append(
                        float(confidence)
                    )

        output: list[PageQualityAssessment] = []
        for page_no, page in sorted(document.pages.items()):
            page_number = int(getattr(page, "page_no", page_no))
            text = "\n".join(text_by_page.get(page_number, ()))
            confidences = confidence_by_page.get(page_number, ())
            mean_confidence = (
                sum(confidences) / len(confidences)
                if confidences
                else None
            )
            gibberish_ratio = cls._gibberish_ratio(text)
            reasons: list[str] = []
            if len(text) < min_text_characters:
                reasons.append("LOW_TEXT_COVERAGE")
            if (
                mean_confidence is not None
                and mean_confidence < min_mean_confidence
            ):
                reasons.append("LOW_EXTRACTION_CONFIDENCE")
            if gibberish_ratio > max_gibberish_ratio:
                reasons.append("EXCESSIVE_GIBBERISH")
            image = getattr(getattr(page, "image", None), "pil_image", None)
            output.append(
                PageQualityAssessment(
                    page_no=page_number,
                    text_characters=len(text),
                    mean_confidence=(
                        round(mean_confidence, 5)
                        if mean_confidence is not None
                        else None
                    ),
                    gibberish_ratio=round(gibberish_ratio, 5),
                    image_available=image is not None,
                    requires_visual=bool(reasons),
                    reasons=tuple(reasons),
                )
            )
        return tuple(output)

    @staticmethod
    def _gibberish_ratio(text: str) -> float:
        if not text:
            return 0.0
        count = sum(
            1
            for character in text
            if character == "\ufffd"
            or (
                ord(character) < 32
                and character not in "\n\r\t"
            )
        )
        isolated_letters = len(re.findall(r"(?:\b[A-Za-z]\s+){4,}", text))
        return (count + isolated_letters) / max(1, len(text))
