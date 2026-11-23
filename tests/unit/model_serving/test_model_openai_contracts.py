"""Model Serving 公开契约与内部语义隔离测试。"""

import unittest
from uuid import UUID

from pydantic import ValidationError

from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_serving.common.openai_router import create_openai_models_router
from model_serving.embedding.schema import OpenAIEmbeddingRequest
from model_serving.llm.schema import OpenAIChatRequest
from model_serving.vlm.schema import OpenAIVLMRequest


CHAT_ID = "019c0000-0000-7000-8000-000000000001"
VLM_ID = "019c0000-0000-7000-8000-000000000002"
EMBED_ID = "019c0000-0000-7000-8000-000000000003"


class ModelOpenAIContractTest(unittest.TestCase):
    def test_chat_model_maps_to_model_id(self):
        request = OpenAIChatRequest(
            model=CHAT_ID,
            messages=[{"role": "user", "content": "你好"}],
        )
        internal = request.to_internal()
        self.assertEqual(UUID(CHAT_ID), internal.model_id)
        self.assertFalse(hasattr(internal, "model"))

    def test_old_call_name_cannot_select_a_model(self):
        with self.assertRaises(ValidationError):
            OpenAIChatRequest(model="chat-prod", messages=[])

    def test_vlm_model_maps_to_model_id(self):
        request = OpenAIVLMRequest(
            model=VLM_ID,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": "描述图片"}],
            }],
        )
        self.assertEqual(UUID(VLM_ID), request.to_internal().model_id)

    def test_embedding_contract_uses_openai_field_names(self):
        request = OpenAIEmbeddingRequest(
            model=EMBED_ID,
            input=["问题一", "问题二"],
        )
        self.assertEqual(UUID(EMBED_ID), request.model)
        self.assertEqual(["问题一", "问题二"], request.input)

    def test_models_endpoint_exposes_only_enabled_model_ids(self):
        class Registry:
            async def list(self, *, category):
                return [
                    {
                        "model_id": "019c0000-0000-7000-8000-000000000001",

                        "status": 1,
                    },
                    {
                        "model_id": "019c0000-0000-7000-8000-000000000003",

                        "status": "ACTIVE",
                    },
                    {
                        "model_id": "019c0000-0000-7000-8000-000000000002",

                        "status": 0,
                    },
                ]

        app = FastAPI()
        app.state.model_registry = Registry()
        app.include_router(create_openai_models_router(category=1))
        payload = TestClient(app).get("/api/v1/models").json()
        self.assertEqual([CHAT_ID, EMBED_ID], [item["id"] for item in payload["data"]])
        self.assertNotIn("model_id", payload["data"][0])


if __name__ == "__main__":
    unittest.main()
