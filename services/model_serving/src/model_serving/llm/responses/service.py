"""LLM 进程内的 Responses 应用服务。"""

from uuid import UUID

from model_serving.common.model_registry import ModelRegistryService
from model_serving.llm.responses.adapter import GenerativeAdapter
from platform_core.contracts import (
    ImageGenerationRequest,
    ImageGenerationResult,
    ResearchRequest,
    ResearchResult,
)


class ResponsesService:
    """能力闸门之后才把请求交给生产或测试适配器。"""

    def __init__(self, *, registry: ModelRegistryService, adapter: GenerativeAdapter):
        self._registry = registry
        self._adapter = adapter

    async def research(self, request: ResearchRequest) -> ResearchResult:
        await self._registry.require_grok_model(request.model_id)
        material = await self._registry.get_connection_material(request.model_id)
        return await self._adapter.research(request, material)

    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        await self._require(request.model_id, "image_generation")
        material = await self._registry.get_connection_material(request.model_id)
        return await self._adapter.generate_image(request, material)

    async def verify_capabilities(
        self,
        model_id: UUID,
        *,
        supports_image_generation: bool,
        actor_id: str,
    ) -> dict:
        """只记录验收标志，不向 OCI 发起 Canary。"""
        return await self._registry.record_capability_verification(
            model_id,
            supports_image_generation=supports_image_generation,
            actor_id=actor_id,
        )

    async def _require(self, model_id: UUID, capability: str) -> None:
        await self._registry.require_verified_capability(model_id, capability=capability)
