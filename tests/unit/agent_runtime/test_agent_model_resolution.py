"""Agent 模型 UUID 绑定与运行快照解析测试。"""

import unittest

from agent_runtime.application.model_resolution import (
    AgentModelCatalogResolver,
)
from agent_runtime.domain.model_bindings import normalize_agent_models
from platform_core.dictionary import ModelCategory
from platform_core.identity import uuid7


class _Client:
    def __init__(self, definitions):
        self._definitions = definitions

    async def get_model(self, model_id):
        row = self._definitions.get(model_id)
        if row is None:
            raise LookupError("模型不存在")
        return row


class AgentModelResolutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_uuid_binding_to_frozen_call_snapshot(self):
        llm_id = uuid7()
        embedding_id = uuid7()
        resolver = AgentModelCatalogResolver(
            {
                ModelCategory.LLM: _Client(
                    {
                        llm_id: {
                            "category": 1,
                            "status": "ACTIVE",
                            "model_params": {"temperature": 0.1},
                        }
                    }
                ),
                ModelCategory.TXT_EMBEDDING: _Client(
                    {
                        embedding_id: {
                            "category": 2,
                            "status": "ACTIVE",
                        }
                    }
                ),
            }
        )

        snapshot = await resolver.resolve(
            {
                "composer_llm": llm_id,
                "memory_embedding": embedding_id,
            }
        )

        self.assertEqual(snapshot["composer_llm"]["model_id"], str(llm_id))
        self.assertEqual(snapshot["memory_embedding"]["model_id"], str(embedding_id))
        self.assertNotIn("served_model_name", snapshot["composer_llm"])
        self.assertEqual(
            len(snapshot["composer_llm"]["config_fingerprint"]), 64
        )

    async def test_rejects_role_category_mismatch(self):
        model_id = uuid7()
        resolver = AgentModelCatalogResolver(
            {
                ModelCategory.LLM: _Client(
                    {
                        model_id: {
                            "category": 2,
                            "status": "ACTIVE",
                        }
                    }
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "LLM"):
            await resolver.resolve({"composer_llm": model_id})

    async def test_resolves_separate_ocr_and_vision_roles(self):
        ocr_id, vision_id = uuid7(), uuid7()
        resolver = AgentModelCatalogResolver({
            ModelCategory.OCR: _Client({ocr_id: {"category": 6, "status": "ACTIVE"}}),
            ModelCategory.VLM: _Client({vision_id: {"category": 5, "status": "ACTIVE", "effective_max_tokens": 4000}}),
        })
        models = await resolver.resolve({"query_ocr": ocr_id, "query_vlm": vision_id})
        self.assertIsNone(models["query_ocr"]["max_tokens"])
        self.assertEqual(models["query_vlm"]["max_tokens"], 4000)
        with self.assertRaises(LookupError):
            await resolver.resolve({"query_ocr": vision_id})

    async def test_archived_model_cannot_create_new_runtime_binding(self):
        model_id = uuid7()
        resolver = AgentModelCatalogResolver({
            ModelCategory.LLM: _Client({
                model_id: {
                    "category": 1,
                    "status": "ARCHIVED",
                }
            })
        })
        with self.assertRaisesRegex(ValueError, "未启用"):
            await resolver.resolve({"composer_llm": model_id})

    def test_configuration_requires_uuidv7_for_every_role(self):
        with self.assertRaisesRegex(ValueError, "UUID"):
            normalize_agent_models(
                {
                    "context_llm": "chat-small",
                    "composer_llm": uuid7(),
                    "memory_llm": uuid7(),
                    "memory_embedding": uuid7(),
                }
            )


if __name__ == "__main__":
    unittest.main()
