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

    async def test_remote_input_chunks_full_text_and_merges_one_vector(self):
        model = OpenAIEmbedding(OpenAIEmbeddingConfig(
            model_name="text-embedding-v4",
            provider="api_qwen",
            max_tokens=6,
            batch_size=10,
            api_key="test-key",
        ))
        model._is_initialized = True
        requested = []

        async def embed_batch(batch, **_):
            requested.extend(batch)
            vectors = {
                "你好": [1.0, 0.0],
                "世界": [0.0, 1.0],
            }
            return [vectors[value] for value in batch], len(batch)

        model._embed_batch = AsyncMock(side_effect=embed_batch)

        response = await model.embed(["你好世界"], batch_size=10)

        self.assertEqual(["你好", "世界"], requested)
        self.assertEqual(1, len(response.data))
        self.assertAlmostEqual(2 ** -0.5, response.data[0].embedding[0])
        self.assertAlmostEqual(2 ** -0.5, response.data[0].embedding[1])
        self.assertEqual(2, response.usage["total_tokens"])

    async def test_qwen_v4_caps_remote_batch_at_provider_limit(self):
        model = OpenAIEmbedding(OpenAIEmbeddingConfig(
            model_name="text-embedding-v4",
            provider="api_qwen",
            max_tokens=8192,
            batch_size=96,
            api_key="test-key",
        ))
        model._is_initialized = True
        batch_sizes = []

        async def embed_batch(batch, **_):
            batch_sizes.append(len(batch))
            return [[0.1] for _ in batch], len(batch)

        model._embed_batch = AsyncMock(side_effect=embed_batch)

        response = await model.embed(
            [f"文本-{index}" for index in range(23)],
            batch_size=96,
        )

        self.assertEqual([10, 10, 3], batch_sizes)
        self.assertEqual(23, len(response.data))

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
