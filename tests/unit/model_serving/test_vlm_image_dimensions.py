"""模型服务只翻译上游明确的图像尺寸拒绝。"""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from openai import AsyncOpenAI

from model_serving.vlm.model.openai_client import OpenAIVLM, OpenAIVLMConfig, VLMImageDimensionsUnsupported
from model_serving.vlm.vlm_service import VLMService
from platform_core.dictionary import VLMProvider

DIMENSION_MESSAGE = ("The image length and width do not meet the model restrictions. "
                     "[height:2 or width:963 must be larger than 10]")


@pytest.mark.parametrize("provider,code,message,expected", [
    (VLMProvider.API_QWEN, "invalid_parameter_error", DIMENSION_MESSAGE, VLMImageDimensionsUnsupported),
    (VLMProvider.API_QWEN, "invalid_parameter_error", "A different invalid parameter", RuntimeError),
    (VLMProvider.API_QWEN, "other_error", DIMENSION_MESSAGE, RuntimeError),
    (VLMProvider.CHATGPT, "invalid_parameter_error", DIMENSION_MESSAGE, RuntimeError),
])
def test_vlm_service_distinguishes_confirmed_dimension_rejection(provider, code, message, expected):
    def respond(request):
        return httpx.Response(400, json={"error": {"code": code, "message": message}}, request=request)

    async def run():
        model = OpenAIVLM(OpenAIVLMConfig(model_name="test-vlm", provider=provider, api_key="fixture"))
        service = VLMService()
        service.get_vlm_model = AsyncMock(return_value=model)
        async with AsyncOpenAI(api_key="fixture", base_url="https://example.invalid/v1", max_retries=0,
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
            model._openai_client = client
            model._is_initialized = True
            with pytest.raises(expected) as exc:
                await service.inference(uuid4(), [{"role": "user", "content": [{"type": "text", "text": "测试"}]}],
                                        expected_revision="a" * 64)
            assert type(exc.value) is expected
            assert service.get_vlm_model.await_args.kwargs["expected_revision"] == "a" * 64

    asyncio.run(run())
