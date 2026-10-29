"""OpenAI 兼容模型的 JSON Mode 与单次请求参数合同。"""

import asyncio
from types import SimpleNamespace

from model_serving.llm.model.openai_client import OpenaiClient, OpenaiLLMConfig


def test_openai_client_forwards_json_object_response_format() -> None:
    async def run() -> None:
        client = OpenaiClient(OpenaiLLMConfig(
            model_name="structured-test-model",
            provider="api_deepseek",
            max_tokens=4096,
            api_endpoint="https://example.invalid/v1",
            api_key="test-key",
        ))
        observed: dict[str, object] = {}

        async def create(**kwargs):
            observed.update(kwargs)
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    message=SimpleNamespace(tool_calls=None),
                )],
            )

        client._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )
        client._is_initialized = True

        await client.chat(
            [{"role": "system", "content": "Return JSON"}],
            stream=False,
            temperature=0,
            response_format={"type": "json_object"},
            timeout=12,
        )

        assert observed["response_format"] == {"type": "json_object"}
        assert observed["stream"] is False
        assert observed["temperature"] == 0
        assert observed["timeout"] == 12

    asyncio.run(run())
