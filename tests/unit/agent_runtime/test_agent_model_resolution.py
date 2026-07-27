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
                            "status": 1,
                            "served_model_name": "chat-prod",
                            "model_params": {"temperature": 0.1},
                        }
                    }
                ),
                ModelCategory.TXT_EMBEDDING: _Client(
                    {
                        embedding_id: {
                            "category": 2,
                            "status": 1,
                            "served_model_name": "embed-prod",
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

        self.assertEqual(
            snapshot["composer_llm"]["model_id"], str(llm_id)
        )
        self.assertEqual(
            snapshot["composer_llm"]["served_model_name"], "chat-prod"
        )
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
                            "status": 1,
                            "served_model_name": "wrong-kind",
                        }
                    }
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "LLM"):
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
