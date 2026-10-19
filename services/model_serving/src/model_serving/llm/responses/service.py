"""LLM 进程内的 Responses 应用服务。"""

from model_serving.common.model_registry import ModelRegistryService
from model_serving.llm.responses.adapter import GenerativeAdapter
from platform_core.contracts import (
    ResearchRequest,
    ResearchResult,
)
from platform_core.dictionary import ModelCategory


class ResponsesService:
    """把已启用的 Grok 研究请求交给生产或测试适配器。"""

    def __init__(self, *, registry: ModelRegistryService, adapter: GenerativeAdapter):
        self._registry = registry
        self._adapter = adapter

    async def research(self, request: ResearchRequest) -> ResearchResult:
        await self._registry.require_grok_model(request.model_id)
        material = await self._registry.get_connection_material(
            request.model_id,
            category=ModelCategory.LLM.value,
        )
        return await self._adapter.research(request, material)
