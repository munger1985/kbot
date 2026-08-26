"""OpenAI 兼容 Embedding 参数测试。"""

import unittest
from unittest.mock import AsyncMock

from model_serving.embedding.model.openai_client import (
    OpenAIEmbedding,
    OpenAIEmbeddingConfig,
)


class OpenAIEmbeddingTest(unittest.IsolatedAsyncioTestCase):
    async def test_qwen_v4_forwards_declared_dimension(self):
        model = OpenAIEmbedding(OpenAIEmbeddingConfig(
            model_name="text-embedding-v4",
            provider="api_qwen",
            max_tokens=1024,
            batch_size=2,
            api_key="test-key",
            dimensions=2048,
        ))
        model._is_initialized = True
        model._embed_batch = AsyncMock(return_value=([[0.1] * 2048], 1))

        response = await model.embed(["测试"], batch_size=1)

        self.assertEqual(2048, len(response.data[0].embedding))
        self.assertEqual(
            2048,
            model._embed_batch.await_args.kwargs["dimensions"],
        )

    async def test_remote_input_uses_configured_utf8_budget(self):
        model = OpenAIEmbedding(OpenAIEmbeddingConfig(
            model_name="text-embedding-v4",
            provider="api_qwen",
            max_tokens=5,
            batch_size=2,
            api_key="test-key",
        ))
        model._is_initialized = True
        model._embed_batch = AsyncMock(return_value=([[0.1]], 1))

        await model.embed(["你好世界"], batch_size=1)

        self.assertEqual(
            ["你"],
            model._embed_batch.await_args.args[0],
        )

    async def test_remote_input_rejects_blank_text(self):
        model = OpenAIEmbedding(OpenAIEmbeddingConfig(
            model_name="text-embedding-v4",
            provider="api_qwen",
            max_tokens=8192,
            batch_size=2,
            api_key="test-key",
        ))
        model._is_initialized = True

        with self.assertRaisesRegex(ValueError, "不能为空"):
            await model.embed(["   "])


if __name__ == "__main__":
    unittest.main()
