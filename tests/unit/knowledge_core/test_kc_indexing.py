import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_core.application.indexing import (
    EmbeddingBatch,
    EmbeddingModelSnapshot,
    _rebind_revision_content,
    retrieval_input_hash,
    validate_embedding_batch,
)
from platform_core.identity import uuid7


class KcIndexingContractTest(unittest.TestCase):
    def setUp(self):
        self.model = EmbeddingModelSnapshot(
            model_id=uuid7(), served_model_name="embed-v2", dimension=3,
            config_fingerprint="a" * 64,
        )

    def test_provider_identity_and_dimension_must_match(self):
        with self.assertRaises(ValueError):
            validate_embedding_batch(
                batch=EmbeddingBatch(vectors=[[1.0, 2.0, 3.0]], served_model_name="other", dimension=3),
                model=self.model, expected_count=1,
            )
        with self.assertRaises(ValueError):
            validate_embedding_batch(
                batch=EmbeddingBatch(vectors=[[1.0, 2.0]], served_model_name="embed-v2", dimension=2),
                model=self.model, expected_count=1,
            )

    def test_hash_is_the_idempotency_key_for_retrieval_text(self):
        self.assertEqual(retrieval_input_hash("same"), retrieval_input_hash("same"))
        self.assertNotEqual(retrieval_input_hash("same"), retrieval_input_hash("changed"))


class KcRevisionContentBindingTest(unittest.IsolatedAsyncioTestCase):
    async def test_ready_reused_documents_move_to_published_revision(self):
        revision_id = uuid7()
        ready_version_id = uuid7()
        member_id = uuid7()
        uow = SimpleNamespace(
            members=SimpleNamespace(list_by_revision=AsyncMock(return_value=[
                SimpleNamespace(
                    member_status="READY",
                    document_version_id=ready_version_id,
                    bundle_revision_document_id=member_id,
                ),
                SimpleNamespace(
                    member_status="FAILED",
                    document_version_id=uuid7(),
                    bundle_revision_document_id=uuid7(),
                ),
            ])),
            evidence=SimpleNamespace(rebind_active_revision=AsyncMock()),
            visual_assets=SimpleNamespace(rebind_active_revision=AsyncMock()),
        )

        await _rebind_revision_content(
            uow, bundle_revision_id=revision_id
        )

        uow.evidence.rebind_active_revision.assert_awaited_once_with(
            document_version_id=ready_version_id,
            bundle_revision_id=revision_id,
            bundle_revision_document_id=member_id,
        )
        uow.visual_assets.rebind_active_revision.assert_awaited_once_with(
            document_version_id=ready_version_id,
            bundle_revision_id=revision_id,
        )


if __name__ == "__main__":
    unittest.main()
