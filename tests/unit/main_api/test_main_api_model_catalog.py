"""Main API 模型目录聚合接口测试。"""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from main_api.api.models import router
from main_api.entrypoints.api import (
    _configure_model_config_clients,
    settings,
)
from platform_core.identity import uuid7


class _ModelClient:
    def __init__(self, rows):
        self.rows = rows

    async def list_models(self):
        return self.rows


class MainApiModelCatalogTest(unittest.TestCase):
    def test_startup_configures_all_model_catalog_clients(self):
        app = FastAPI()

        _configure_model_config_clients(app)

        clients = app.state.model_config_clients
        self.assertEqual(4, len(clients))
        self.assertEqual(
            tuple(
                dependency.base_url.rstrip("/").replace(
                    "://0.0.0.0", "://127.0.0.1"
                )
                for dependency in (
                    settings.model_embedding,
                    settings.model_llm,
                    settings.model_visual,
                    settings.model_vlm,
                )
            ),
            tuple(client.base_url for client in clients),
        )

    def test_catalog_returns_503_when_clients_are_not_initialized(self):
        app = FastAPI()
        app.include_router(router)

        response = TestClient(app).get("/api/v1/model-catalog")

        self.assertEqual(503, response.status_code)
        self.assertEqual(
            "MODEL_CATALOG_UNAVAILABLE",
            response.json()["detail"]["code"],
        )

    def test_catalog_returns_only_enabled_models_and_keeps_uuid(self):
        llm_id = uuid7()
        embedding_id = uuid7()
        app = FastAPI()
        app.state.model_config_clients = (
            _ModelClient(
                [
                    {
                        "model_id": str(llm_id),
                        "served_model_name": "chat-prod",
                        "display_name": "Chat Prod",
                        "category": 1,
                        "provider": "api_qwen",
                        "status": "ACTIVE",
                    },
                    {
                        "model_id": str(uuid7()),
                        "served_model_name": "chat-disabled",
                        "display_name": "Chat Disabled",
                        "category": 1,
                        "provider": "api_qwen",
                        "status": 0,
                    },
                ]
            ),
            _ModelClient(
                [
                    {
                        "model_id": str(embedding_id),
                        "served_model_name": "embed-prod",
                        "display_name": "Embedding Prod",
                        "category": 2,
                        "provider": "local_qwen",
                        "status": "ACTIVE",
                        "model_params": {"embedding_dimension": 2048},
                    }
                ]
            ),
        )
        app.include_router(router)

        response = TestClient(app).get("/api/v1/model-catalog")

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual(
            [str(llm_id), str(embedding_id)],
            [row["model_id"] for row in payload],
        )
        self.assertEqual([1, 2], [row["category"] for row in payload])


if __name__ == "__main__":
    unittest.main()
