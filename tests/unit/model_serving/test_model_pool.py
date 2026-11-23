"""模型池并发加载与UUID 语义测试。"""

import asyncio
import unittest
from types import SimpleNamespace
from typing import Any

from model_serving.common.model_pool import BaseModelPool
from model_serving.common.frozen_binding import FrozenModelConflict, frozen_revision
from model_serving.embedding.model.openai_client import OpenAIEmbeddingConfig
from model_serving.embedding.model_pool import EmbeddingModelPool


CHAT_ID = "019c0000-0000-7000-8000-000000000001"
EMBEDDING_ID = "019c0000-0000-7000-8000-000000000002"
MISSING_ID = "019c0000-0000-7000-8000-000000000003"


class _Pool(BaseModelPool[dict]):
    def __init__(self):
        super().__init__()
        self.fetch_count: dict[str, int] = {}
        self.start_count: dict[str, int] = {}

    async def _fetch_model_data(self, model_id: str) -> dict[str, Any]:
        self.fetch_count[model_id] = (
            self.fetch_count.get(model_id, 0) + 1
        )
        await asyncio.sleep(0)
        return {
            "model_id": model_id,
            "provider_model_name": f"provider-{model_id}",
        }

    async def _start_model(
        self, model_id: str, model_data: dict[str, Any],
    ) -> dict:
        self.start_count[model_id] = (
            self.start_count.get(model_id, 0) + 1
        )
        await asyncio.sleep(0)
        return dict(model_data)

    async def _shutdown_model_instance(self, model: dict) -> None:
        return None

    async def _perform_model_health_check(
        self, model_id: str, model: dict,
    ) -> None:
        return None

    def _get_model_category(self) -> int:
        return 2


class ModelPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_frozen_models_check_catalog_and_loaded_instance(self):
        pool = _Pool()
        identity = EMBEDDING_ID
        kwargs = {}
        lookup = pool._fetch_model_data
        old = await pool.load_model(identity, **kwargs)
        old_revision = frozen_revision(old)
        self.assertIs(old, await pool.load_model(identity, expected_revision=old_revision, **kwargs))

        async def changed(model_identity, **scope):
            return {**await lookup(model_identity, **scope), "definition_version": 2}

        pool._fetch_model_data = changed
        with self.assertRaises(FrozenModelConflict):
            await pool.load_model(identity, expected_revision=old_revision, **kwargs)
        new_revision = frozen_revision(await changed(identity, **kwargs))
        with self.assertRaises(FrozenModelConflict):
            await pool.load_model(identity, expected_revision=new_revision, **kwargs)
        await pool.unload_model(identity)
        new = await pool.load_model(identity, expected_revision=new_revision, **kwargs)
        self.assertIsNot(old, new)
        self.assertEqual(2, new["definition_version"])

    def test_api_embedding_uses_catalogue_dimension(self):
        config = EmbeddingModelPool()._build_config(
            "qwen-v4",
            "api_qwen",
            {
                "provider_model_name": "text-embedding-v4",
                "api_key": "test-key",
                "api_endpoint": "https://example.invalid/v1",
                "model_params": {"embedding_dimension": 2048},
            },
            SimpleNamespace(max_tokens=1024, timeout=30),
        )

        self.assertIsInstance(config, OpenAIEmbeddingConfig)
        self.assertEqual(2048, config.dimensions)

    async def test_same_model_is_cold_started_only_once(self):
        pool = _Pool()
        models = await asyncio.gather(
            *(pool.load_model(EMBEDDING_ID) for _ in range(10))
        )
        self.assertTrue(all(model is models[0] for model in models))
        self.assertEqual(1, pool.fetch_count[EMBEDDING_ID])
        self.assertEqual(1, pool.start_count[EMBEDDING_ID])

    async def test_model_id_and_provider_name_stay_separate(self):
        pool = _Pool()
        model = await pool.load_model(CHAT_ID)
        self.assertEqual(CHAT_ID, model["model_id"])
        self.assertEqual(f"provider-{CHAT_ID}", model["provider_model_name"])

    async def test_invalid_uuid_is_rejected(self):
        pool = _Pool()
        with self.assertRaises(ValueError):
            await pool.load_model(" ")

    async def test_failed_load_releases_lifecycle_lock(self):
        pool = _Pool()

        async def fail_fetch(model_id: str):
            raise RuntimeError(f"模型不存在：{model_id}")

        pool._fetch_model_data = fail_fetch
        with self.assertRaisesRegex(RuntimeError, "模型不存在"):
            await pool.load_model(MISSING_ID)
        self.assertNotIn(MISSING_ID, pool._model_locks)

    async def test_invalidation_makes_next_request_load_new_configuration(self):
        pool = _Pool()
        revision = {"value": "v1"}

        async def fetch(model_id: str):
            return {
                "model_id": model_id,
                "provider_model_name": revision["value"],
            }

        pool._fetch_model_data = fetch
        first = await pool.load_model(CHAT_ID)
        revision["value"] = "v2"
        await pool.unload_model(CHAT_ID)
        second = await pool.load_model(CHAT_ID)
        self.assertEqual("v1", first["provider_model_name"])
        self.assertEqual("v2", second["provider_model_name"])

    async def test_new_process_loads_catalogue_configuration(self):
        first_pool = _Pool()
        await first_pool.load_model(CHAT_ID)
        await first_pool.shutdown()
        restarted_pool = _Pool()
        model = await restarted_pool.load_model(CHAT_ID)
        self.assertEqual(f"provider-{CHAT_ID}", model["provider_model_name"])


if __name__ == "__main__":
    unittest.main()
