"""独立图像生成服务的模型类别边界。"""

import unittest
from unittest.mock import AsyncMock

from model_serving.image.adapter import ImageModelAdapter
from model_serving.image.service import ImageService
from platform_core.contracts import ImageGenerationRequest, ImageGenerationResult
from platform_core.dictionary import ModelCategory
from platform_core.identity import uuid7


class ImageServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_provider_identity_maps_to_oci_transport(self):
        adapter = ImageModelAdapter()
        adapter._oci.generate_image = AsyncMock(return_value=ImageGenerationResult(
            status="FAILED",
            error_code="PROVIDER_UNAVAILABLE",
        ))
        request = ImageGenerationRequest(
            model_id=uuid7(),
            prompt="a blue cube",
            trace_id="trace-image-provider",
        )

        await adapter.generate_image(
            request,
            {"provider": "api_oci_image", "provider_model_name": "grok-4.6"},
        )

        passed = adapter._oci.generate_image.await_args.args[1]
        self.assertEqual("oci", passed["provider"])

    async def test_generation_resolves_only_image_generation_category(self):
        model_id = uuid7()
        registry = AsyncMock()
        registry.get_connection_material.return_value = {
            "provider": "api_oci_image",
        }
        adapter = AsyncMock()
        adapter.generate_image.return_value = ImageGenerationResult(
            status="FAILED",
            error_code="PROVIDER_UNAVAILABLE",
        )
        service = ImageService(registry=registry, adapter=adapter)
        request = ImageGenerationRequest(
            model_id=model_id,
            prompt="a blue cube",
            trace_id="trace-image",
        )

        await service.generate_image(request)

        registry.get_connection_material.assert_awaited_once_with(
            model_id,
            category=ModelCategory.IMAGE_GENERATION.value,
        )
        adapter.generate_image.assert_awaited_once_with(
            request,
            {"provider": "api_oci_image"},
        )


if __name__ == "__main__":
    unittest.main()
