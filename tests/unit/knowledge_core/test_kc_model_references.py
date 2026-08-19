"""Knowledge Core Collection 与索引模型引用反查测试。"""

from types import SimpleNamespace
import unittest

from knowledge_core.repositories.model_reference_repo import (
    ModelReferenceRepository,
)
from platform_core.identity import uuid7


class _Result:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return self._values


class _Session:
    def __init__(self, results):
        self._results = iter(results)

    async def execute(self, statement):
        del statement
        return _Result(next(self._results))


class ModelReferenceRepositoryTest(unittest.IsolatedAsyncioTestCase):
    async def test_lists_collection_and_persisted_index_references(self):
        model_id = uuid7()
        collection_id = uuid7()
        collection = SimpleNamespace(
            collection_id=collection_id,
            domain_id=20,
            display_name="产品知识库",
            status="ACTIVE",
            models_json={
                "embedding": str(model_id),
            },
        )
        repository = ModelReferenceRepository(
            _Session(
                [
                    [collection],
                    [collection_id],
                    [collection_id],
                    [],
                ]
            )
        )

        references = await repository.list_by_model(model_id=model_id)

        self.assertEqual(3, len(references))
        self.assertEqual(
            {"collection", "index_profile"},
            {item["resource_type"] for item in references},
        )
        self.assertEqual(
            {
                "embedding",
                "evidence_embedding",
                "discovery_embedding",
            },
            {item["binding_role"] for item in references},
        )
        self.assertTrue(
            all(item["domain_id"] == 20 for item in references)
        )

    async def test_unreferenced_model_returns_empty_list(self):
        collection = SimpleNamespace(
            collection_id=uuid7(),
            domain_id=20,
            display_name="产品知识库",
            status="ACTIVE",
            models_json={"embedding": str(uuid7())},
        )
        repository = ModelReferenceRepository(
            _Session([[collection], [], [], []])
        )

        self.assertEqual([], await repository.list_by_model(model_id=uuid7()))


if __name__ == "__main__":
    unittest.main()
