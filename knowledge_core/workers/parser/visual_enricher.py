"""Optional, policy-controlled visual description enhancement."""

import asyncio

from docling_core.types.doc.document import DescriptionAnnotation
from loguru import logger

from platform_clients import AIModelClient


class KcVisualEnricher:
    def __init__(self, client_factory=AIModelClient):
        self._client_factory = client_factory

    async def enrich(
        self, document, *, model_name: str | None, prompt: str,
        max_concurrency: int = 2,
    ) -> int:
        if not model_name:
            return 0
        client = self._client_factory()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def enrich_picture(picture) -> bool:
            existing = [
                annotation for annotation in getattr(picture, "annotations", ())
                if isinstance(annotation, DescriptionAnnotation)
                and annotation.text and any(
                    marker in str(annotation.provenance).lower() for marker in ("vlm", "visual")
                )
            ]
            image = getattr(getattr(picture, "image", None), "pil_image", None)
            if existing or image is None:
                return False
            async with semaphore:
                description = await client.get_vlm_answer(model_name, image, prompt=prompt)
            if not description.strip():
                return False
            picture.annotations.append(DescriptionAnnotation(
                text=description.strip(), provenance=f"vlm_kc_v2:{model_name}",
            ))
            return True

        results = await asyncio.gather(
            *(enrich_picture(picture) for picture in document.pictures),
            return_exceptions=True,
        )
        for failure in (result for result in results if isinstance(result, Exception)):
            logger.warning("KC 视觉描述生成失败：{}", failure)
        return sum(result is True for result in results)
