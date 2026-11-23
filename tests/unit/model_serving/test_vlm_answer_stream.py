"""图文问答的流式用量尾帧回归测试。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from model_serving.vlm.vlm_service import VLMService


@pytest.mark.parametrize("with_usage", [False, True])
def test_vlm_stream_completes_with_optional_usage_frame(with_usage):
    async def provider():
        yield SimpleNamespace(
            usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="合成图文回答"))])
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
        if with_usage:
            yield SimpleNamespace(
                usage=SimpleNamespace(total_tokens=5, prompt_tokens=3, completion_tokens=2), choices=[])

    async def run():
        service = VLMService()
        model = SimpleNamespace(inference=AsyncMock(return_value=provider()))
        service.get_vlm_model = AsyncMock(return_value=model)
        stream = await service.inference(uuid4(), [], stream=True)
        assert await anext(stream) == "合成图文回答"
        remaining = [part async for part in stream]
        if with_usage:
            assert len(remaining) == 1
            assert json.loads(remaining[0].removeprefix("\n\n=== USAGE ===\n")) == {
                "total_tokens": 5, "prompt_tokens": 3, "completion_tokens": 2,
            }
        else:
            assert remaining == []

    asyncio.run(run())
