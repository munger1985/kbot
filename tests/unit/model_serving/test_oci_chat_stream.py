"""OCI 原生流式帧经过模型服务路由后须满足 ChatCompletionChunk 合同。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from model_serving.entrypoints.llm import handle_chat_completions
from model_serving.llm.schema import ChatRequest
from platform_core.dictionary import LLMProvider


async def open_stream(chunks):
    async def provider():
        for chunk in chunks:
            yield chunk

    model_id = uuid4()
    request = ChatRequest(model_id=model_id, messages=[{"role": "user", "content": "合成提问"}], stream=True)
    service = SimpleNamespace(
        get_llm_model=AsyncMock(return_value=SimpleNamespace(config=SimpleNamespace(
            provider=LLMProvider.OCI.value, max_tokens=1234,
        ))),
        chat=AsyncMock(return_value=provider()),
    )
    response = await handle_chat_completions(request, service=service)
    return response.body_iterator, str(model_id)


def decode_frame(frame):
    data = frame.removeprefix("data: ").strip()
    return data if data == "[DONE]" else json.loads(data)


def test_oci_stream_delivers_native_text_then_finish_and_independent_usage():
    async def run():
        stream, model_id = await open_stream([
            {"message": {"role": "ASSISTANT", "content": [
                {"type": "THINKING", "text": "合成推理内容"},
                {"type": "TEXT", "text": "荧光光谱法"},
            ]}, "index": 0, "pad": ""},
            {"apiFormat": "GENERIC", "choices": [{"index": 0, "message": {
                "role": "ASSISTANT", "content": [{"type": "TEXT", "text": "用于分析荧光。[D1]"}],
            }}]},
            {"index": 0, "message": None, "finishReason": "STOP"},
            {"usage": {"promptTokens": 21, "completionTokens": 9, "totalTokens": 30}},
        ])
        first = decode_frame(await anext(stream))
        assert first["model"] == model_id
        assert first["choices"][0]["delta"] == {"content": "荧光光谱法"}
        text, finish, usage, done = [decode_frame(frame) async for frame in stream]
        assert text["choices"][0]["delta"]["content"] == "用于分析荧光。[D1]"
        assert finish["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        assert usage["choices"] == []
        assert usage["usage"] == {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30}
        assert done == "[DONE]"

    asyncio.run(run())


def test_oci_cohere_finish_reason_is_not_lost():
    async def run():
        stream, _ = await open_stream([
            {"apiFormat": "COHERE", "text": "合成内容", "finishReason": "MAX_TOKENS"},
        ])
        content, done = [decode_frame(frame) async for frame in stream]
        assert content["choices"][0] == {
            "index": 0, "delta": {"content": "合成内容"}, "finish_reason": "length",
        }
        assert done == "[DONE]"

    asyncio.run(run())


@pytest.mark.parametrize("bad_frame", [{"error": "不透传上游错误内容"}, {"unexpected": True}, {}])
def test_oci_invalid_or_error_frame_does_not_become_success(bad_frame):
    async def run():
        stream, _ = await open_stream([bad_frame])
        error, done = [decode_frame(frame) async for frame in stream]
        assert error == {"error": "MODEL_STREAM_FAILED"}
        assert done == "[DONE]"

    asyncio.run(run())
