"""仅用于测试的生成适配器，不得接入生产路径。"""

from typing import Any

from model_serving.llm.responses.errors import GenerativeAdapterError
from platform_core.contracts import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ResearchRequest,
    ResearchResult,
)


class FakeGenerativeAdapter:
    """由单测注入的固定结果适配器。"""

    def __init__(
        self,
        *,
        research_result: ResearchResult | None = None,
        image_result: ImageGenerationResult | None = None,
        error: GenerativeAdapterError | None = None,
    ):
        self.research_result = research_result
        self.image_result = image_result
        self.error = error
        self.calls: list[tuple[str, Any]] = []

    async def research(
        self, request: ResearchRequest, material: dict[str, Any],
    ) -> ResearchResult:
        self.calls.append(("research", request))
        if self.error is not None:
            raise self.error
        if self.research_result is None:
            raise GenerativeAdapterError("PROVIDER_UNAVAILABLE", "测试适配器未配置研究结果")
        return self.research_result

    async def generate_image(
        self, request: ImageGenerationRequest, material: dict[str, Any],
    ) -> ImageGenerationResult:
        self.calls.append(("image", request))
        if self.error is not None:
            raise self.error
        if self.image_result is None:
            raise GenerativeAdapterError("PROVIDER_UNAVAILABLE", "测试适配器未配置文生图结果")
        return self.image_result
