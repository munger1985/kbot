import unittest

from knowledge_core.application.indexing import EmbeddingBatch, EmbeddingModelSnapshot
from knowledge_core.application.query_embeddings import CollectionQueryEmbeddingProvider
from platform_core.identity import uuid7


MODEL_A = uuid7()
MODEL_B = uuid7()


class _Collection:
    def __init__(self, model_id):
        self.embedding_model_id = model_id


class _Collections:
    async def get_by_id(self, *, collection_id):
        return {
            1: _Collection(MODEL_A),
            2: _Collection(MODEL_A),
            3: _Collection(MODEL_B),
        }.get(collection_id)


class _Uow:
    collections = _Collections()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _Gateway:
    def __init__(self):
        self.calls = []

    async def embed_texts(self, *, served_model_name, texts, is_query=False):
        self.calls.append((served_model_name, list(texts), is_query))
        return EmbeddingBatch(vectors=[[float(len(self.calls))]], served_model_name=served_model_name, dimension=1)


class QueryEmbeddingProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_groups_collections_by_bound_model_and_uses_query_mode(self):
        gateway = _Gateway()

        async def resolve(model_id):
            return EmbeddingModelSnapshot(model_id, f"model-{model_id}", 1, "a" * 64)

        provider = CollectionQueryEmbeddingProvider(
            uow_factory=_Uow, embedding_gateway=gateway, model_resolver=resolve,
        )
        vectors = await provider.embed_for_collections(query="问题", collection_ids=(3, 1, 2))
        self.assertEqual(
            {call[0] for call in gateway.calls},
            {f"model-{MODEL_A}", f"model-{MODEL_B}"},
        )
        self.assertTrue(all(call[2] for call in gateway.calls))
        self.assertEqual(set(vectors), {1, 2, 3})
        self.assertEqual(vectors[1], vectors[2])


if __name__ == "__main__":
    unittest.main()
