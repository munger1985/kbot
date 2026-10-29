"""交互剩余期限必须传递到 Provider，截止后不继续生成长回答。"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from model_serving.llm.llm_service import LLMService


def test_nonstream_timeout_cancels_provider_without_changing_shared_config():
    async def exercise():
        cancelled = asyncio.Event()
        async def generate(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        model = SimpleNamespace(config=SimpleNamespace(provider="api_deepseek", max_tokens=16384), chat=AsyncMock(side_effect=generate))
        service = LLMService()
        service.get_llm_model = AsyncMock(return_value=model)
        with pytest.raises(TimeoutError):
            await service.chat(served_model_name="test", messages="JSON", timeout=0.02)
        assert cancelled.is_set() and model.config.max_tokens == 16384
        assert model.chat.await_args.kwargs["timeout"] == 0.02
        assert "max_tokens" not in model.chat.await_args.kwargs
    asyncio.run(exercise())
