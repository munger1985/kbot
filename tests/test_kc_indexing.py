import unittest

from knowledge_core.application.indexing import (
    EmbeddingBatch,
    EmbeddingModelSnapshot,
    retrieval_input_hash,
    validate_embedding_batch,
)


class KcIndexingContractTest(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingModelSnapshot(
            model_id=7, model_key="embed-v2", dimension=3,
            config_fingerprint="a" * 64,
        )

    def test_provider_identity_and_dimension_must_match(self):
        with self.assertRaises(ValueError):
            validate_embedding_batch(
                batch=EmbeddingBatch(vectors=[[1.0, 2.0, 3.0]], model_key="other", dimension=3),
                model=self.model, expected_count=1,
            )
        with self.assertRaises(ValueError):
            validate_embedding_batch(
                batch=EmbeddingBatch(vectors=[[1.0, 2.0]], model_key="embed-v2", dimension=2),
                model=self.model, expected_count=1,
            )

    def test_hash_is_the_idempotency_key_for_retrieval_text(self):
        self.assertEqual(retrieval_input_hash("same"), retrieval_input_hash("same"))
        self.assertNotEqual(retrieval_input_hash("same"), retrieval_input_hash("changed"))


if __name__ == "__main__":
    unittest.main()
