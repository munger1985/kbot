"""Model Serving 公开契约与内部语义隔离测试。"""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_serving.common.openai_router import create_openai_models_router
from model_serving.embedding.schema import OpenAIEmbeddingRequest
from model_serving.llm.schema import OpenAIChatRequest
from model_serving.vlm.schema import OpenAIVLMRequest


class ModelOpenAIContractTest(unittest.TestCase):
    def test_chat_model_maps_to_served_model_name(self):
        request = OpenAIChatRequest(
            model="chat-prod",
            messages=[{"role": "user", "content": "你好"}],
        )
        internal = request.to_internal()
        self.assertEqual("chat-prod", internal.served_model_name)
        self.assertFalse(hasattr(internal, "model"))

    def test_vlm_model_maps_to_served_model_name(self):
        request = OpenAIVLMRequest(
            model="vlm-prod",
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "描述图片"}],
            }],
        )
        self.assertEqual("vlm-prod", request.to_internal().served_model_name)

    def test_embedding_contract_uses_openai_field_names(self):
        request = OpenAIEmbeddingRequest(
            model="embed-prod",
            input=["问题一", "问题二"],
        )
        self.assertEqual("embed-prod", request.model)
        self.assertEqual(["问题一", "问题二"], request.input)

    def test_models_endpoint_exposes_only_enabled_served_names(self):
        class Registry:
            async def list(self, *, category):
                return [
                    {
                        "model_id": "019c0000-0000-7000-8000-000000000001",
                        "served_model_name": "chat-prod",
                        "status": 1,
                    },
                    {
                        "model_id": "019c0000-0000-7000-8000-000000000002",
                        "served_model_name": "chat-disabled",
                        "status": 0,
                    },
                ]

        app = FastAPI()
        app.state.model_registry = Registry()
        app.include_router(create_openai_models_router(category=1))
        payload = TestClient(app).get("/api/v1/models").json()
        self.assertEqual(["chat-prod"], [item["id"] for item in payload["data"]])
        self.assertNotIn("model_id", payload["data"][0])


if __name__ == "__main__":
    unittest.main()
