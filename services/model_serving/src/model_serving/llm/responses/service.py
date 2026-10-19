"""LLM 进程内的 Responses 应用服务。"""

from uuid import UUID

from model_serving.common.model_registry import ModelRegistryService
from model_serving.llm.responses.adapter import GenerativeAdapter
from model_serving.llm.responses.errors import GenerativeAdapterError
from platform_core.contracts import (
    AuthContext,
    ImageGenerationRequest,
    ImageGenerationResult,
    ResearchRequest,
    ResearchResult,
)
from platform_core.dictionary import ModelCategory


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

    async def verify_image_generation(
        self,
        model_id: UUID,
        *,
        expected_row_version: int,
        actor_id: str,
        trace_id: str,
        auth_context: AuthContext,
    ) -> dict:
        """执行真实文生图 Canary，成功取得图片后才记录能力。"""
        material = await self._registry.get_capability_verification_material(
            model_id,
            expected_row_version=expected_row_version,
            category=ModelCategory.LLM.value,
        )
        result = await self._adapter.generate_image(
            ImageGenerationRequest(
                model_id=model_id,
                prompt="生成一个白色背景、居中的蓝色正方形测试图标。",
                aspect_ratio="1:1",
                count=1,
                trace_id=trace_id,
            ),
            material,
        )
        if result.status != "COMPLETED" or not result.artifacts:
            code = result.error_code or "PROVIDER_UNAVAILABLE"
            raise GenerativeAdapterError(
                code,
                result.error_message or "文生图 Canary 未返回有效图片",
                status_code=422 if code == "CONTENT_REJECTED" else 502,
            )
        return await self._registry.record_capability_verification(
            model_id,
            supports_image_generation=True,
            expected_row_version=expected_row_version,
            actor_id=actor_id,
            auth_context=auth_context,
        )

    async def _require(self, model_id: UUID, capability: str) -> None:
        await self._registry.require_verified_capability(model_id, capability=capability)
