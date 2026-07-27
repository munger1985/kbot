"""模型池并发加载与服务名语义测试。"""

import asyncio
import unittest
from typing import Any

from model_serving.common.model_pool import BaseModelPool


class _Pool(BaseModelPool[dict]):
    def __init__(self):
        super().__init__()
        self.fetch_count: dict[str, int] = {}
        self.start_count: dict[str, int] = {}

    async def _fetch_model_data(self, served_model_name: str) -> dict[str, Any]:
        self.fetch_count[served_model_name] = (
            self.fetch_count.get(served_model_name, 0) + 1
        )
        await asyncio.sleep(0)
        return {
            "served_model_name": served_model_name,
            "provider_model_name": f"provider-{served_model_name}",
        }

    async def _start_model(
        self, served_model_name: str, model_data: dict[str, Any],
    ) -> dict:
        self.start_count[served_model_name] = (
            self.start_count.get(served_model_name, 0) + 1
        )
        await asyncio.sleep(0)
        return dict(model_data)

    async def _shutdown_model_instance(self, model: dict) -> None:
        return None

    async def _perform_model_health_check(
        self, served_model_name: str, model: dict,
    ) -> None:
        return None

    def _get_model_category(self) -> int:
        return 2


class ModelPoolTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_model_is_cold_started_only_once(self):
        pool = _Pool()
        models = await asyncio.gather(
            *(pool.load_model("embed-prod") for _ in range(10))
        )
        self.assertTrue(all(model is models[0] for model in models))
        self.assertEqual(1, pool.fetch_count["embed-prod"])
        self.assertEqual(1, pool.start_count["embed-prod"])

    async def test_served_name_and_provider_name_stay_separate(self):
        pool = _Pool()
        model = await pool.load_model("chat-prod")
        self.assertEqual("chat-prod", model["served_model_name"])
        self.assertEqual("provider-chat-prod", model["provider_model_name"])

    async def test_blank_served_name_is_rejected(self):
        pool = _Pool()
        with self.assertRaisesRegex(ValueError, "served_model_name"):
            await pool.load_model(" ")

    async def test_failed_load_releases_lifecycle_lock(self):
        pool = _Pool()

        async def fail_fetch(served_model_name: str):
            raise RuntimeError(f"模型不存在：{served_model_name}")

        pool._fetch_model_data = fail_fetch
        with self.assertRaisesRegex(RuntimeError, "模型不存在"):
            await pool.load_model("missing-model")
        self.assertNotIn("missing-model", pool._model_locks)


if __name__ == "__main__":
    unittest.main()
