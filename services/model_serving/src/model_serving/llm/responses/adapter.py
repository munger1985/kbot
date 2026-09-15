"""生成能力适配器协议。"""

from typing import Any, Protocol

from platform_core.contracts import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ResearchRequest,
    ResearchResult,
)


class GenerativeAdapter(Protocol):
    async def research(
        self, request: ResearchRequest, material: dict[str, Any],
    ) -> ResearchResult:
        ...

    async def generate_image(
        self, request: ImageGenerationRequest, material: dict[str, Any],
    ) -> ImageGenerationResult:
        ...
