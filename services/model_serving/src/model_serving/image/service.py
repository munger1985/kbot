"""图像生成与编辑应用服务。"""

from model_serving.common.model_registry import ModelRegistryService
from platform_core.contracts import ImageGenerationRequest, ImageGenerationResult
from platform_core.dictionary import ModelCategory

from .adapter import ImageModelAdapter


class ImageService:
    """只使用已启用的图像生成类别模型执行真实调用。"""

    def __init__(
        self, *, registry: ModelRegistryService, adapter: ImageModelAdapter,
    ) -> None:
        self._registry = registry
        self._adapter = adapter

    async def generate_image(
        self, request: ImageGenerationRequest,
    ) -> ImageGenerationResult:
        material = await self._registry.get_connection_material(
            request.model_id,
            category=ModelCategory.IMAGE_GENERATION.value,
        )
        return await self._adapter.generate_image(request, material)
