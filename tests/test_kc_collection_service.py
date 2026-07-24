"""Unit tests for Collection root creation without an Oracle dependency."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from platform_core.identity import uuid7

from knowledge_core.application.collections import (
    BindAgentCollectionCommand,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    CreateCollectionCommand,
    KnowledgeCoreBindingService,
    KnowledgeCoreCollectionService,
    UpdateCollectionGenerationModelsCommand,
)


class FakeCollectionRepository:
    def __init__(self, existing=None):
        self._existing = existing
        self.added = None

    async def get_by_scope_key(self, **kwargs):
        return self._existing

    async def add(self, collection):
        collection.collection_id = 101
        self.added = collection
        return collection

    async def list_by_scope(self, **kwargs):
        del kwargs
        return [] if self._existing is None else [self._existing]


class FakeBindingRepository:
    def __init__(self, existing=None):
        self._existing = existing
        self.added = None

    async def get_by_consumer_collection(self, **kwargs):
        return self._existing

    async def add(self, binding):
        binding.binding_id = 201
        self.added = binding
        return binding


class FakeUnitOfWork:
    def __init__(self, repository, binding_repository=None):
        self.collections = repository
        self.bindings = binding_repository
        self.session = SimpleNamespace(flush=AsyncMock())
        self.commit = AsyncMock()
        self.jobs = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class KnowledgeCoreCollectionServiceTest(unittest.IsolatedAsyncioTestCase):
    def _service(self, repository):
        uow = FakeUnitOfWork(repository)
        service = KnowledgeCoreCollectionService(app_id=112, uow_factory=lambda: uow)
        return service, uow

    def _binding_service(self, collection_repository, binding_repository):
        uow = FakeUnitOfWork(collection_repository, binding_repository)
        service = KnowledgeCoreBindingService(app_id=112, uow_factory=lambda: uow)
        return service, uow

    async def test_creates_collection_with_server_injected_app_id(self):
        repository = FakeCollectionRepository()
        service, uow = self._service(repository)
        embedding_model_id = uuid7()

        collection = await service.create(
            CreateCollectionCommand(
                domain_id=8,
                collection_key="assets",
                display_name="Asset Knowledge",
                parser_llm_model_id=uuid7(),
                parser_vlm_model_id=uuid7(),
                retrieval_llm_model_id=uuid7(),
                embedding_model_id=embedding_model_id,
                metadata={"source": "km"},
                actor_id="user:7",
            )
        )

        self.assertEqual(101, collection.collection_id)
        self.assertEqual(112, repository.added.app_id)
        self.assertEqual(8, repository.added.domain_id)
        self.assertEqual(embedding_model_id, repository.added.embedding_model_id)
        self.assertEqual("ACTIVE", repository.added.status)
        uow.commit.assert_awaited_once()

    async def test_rejects_existing_immutable_collection_key(self):
        repository = FakeCollectionRepository(existing=object())
        service, uow = self._service(repository)

        with self.assertRaises(CollectionAlreadyExistsError):
            await service.create(
                CreateCollectionCommand(
                    domain_id=8,
                    collection_key="assets",
                    display_name="Asset Knowledge",
                    parser_llm_model_id=uuid7(),
                    parser_vlm_model_id=None,
                    retrieval_llm_model_id=uuid7(),
                    embedding_model_id=uuid7(),
                )
            )

        self.assertIsNone(repository.added)
        uow.commit.assert_not_awaited()

    async def test_list_returns_session_independent_snapshots(self):
        collection = SimpleNamespace(
            collection_id=uuid7(),
            app_id=112,
            domain_id=8,
            collection_key="assets",
            display_name="Asset Knowledge",
            description=None,
            parser_llm_model_id=uuid7(),
            parser_vlm_model_id=None,
            retrieval_llm_model_id=uuid7(),
            embedding_model_id=uuid7(),
            visual_embedding_model_id=None,
            status="ACTIVE",
            default_security_level=1,
            metadata_json={"source": "test"},
        )
        service, _ = self._service(
            FakeCollectionRepository(existing=collection)
        )

        result = await service.list(domain_id=8)
        collection.display_name = "数据库实体已变化"

        self.assertEqual(1, len(result))
        self.assertEqual("Asset Knowledge", result[0].display_name)
        self.assertEqual({"source": "test"}, result[0].metadata_json)


    async def test_rejects_collection_without_a_valid_embedding_model(self):
        repository = FakeCollectionRepository()
        service, uow = self._service(repository)

        with self.assertRaisesRegex(ValueError, "embedding_model_id"):
            await service.create(CreateCollectionCommand(
                domain_id=8,
                collection_key="assets",
                display_name="Asset Knowledge",
                parser_llm_model_id=uuid7(),
                parser_vlm_model_id=None,
                retrieval_llm_model_id=uuid7(),
                embedding_model_id="not-a-uuid",
            ))

        self.assertIsNone(repository.added)
        uow.commit.assert_not_awaited()

    async def test_updates_only_generation_models(self):
        original_embedding = uuid7()
        collection = SimpleNamespace(
            parser_llm_model_id=uuid7(),
            parser_vlm_model_id=None,
            retrieval_llm_model_id=uuid7(),
            embedding_model_id=original_embedding,
            visual_embedding_model_id=uuid7(),
            updated_by=None,
        )
        service, uow = self._service(
            FakeCollectionRepository(existing=collection)
        )
        parser_llm = uuid7()
        parser_vlm = uuid7()
        retrieval_llm = uuid7()

        updated = await service.update_generation_models(
            UpdateCollectionGenerationModelsCommand(
                domain_id=8,
                collection_key="assets",
                parser_llm_model_id=parser_llm,
                parser_vlm_model_id=parser_vlm,
                retrieval_llm_model_id=retrieval_llm,
                actor_id="user:7",
            )
        )

        self.assertEqual(parser_llm, updated.parser_llm_model_id)
        self.assertEqual(parser_vlm, updated.parser_vlm_model_id)
        self.assertEqual(retrieval_llm, updated.retrieval_llm_model_id)
        self.assertEqual(original_embedding, updated.embedding_model_id)
        uow.commit.assert_awaited_once()

    async def test_binds_agent_to_collection_without_retrieval_weights(self):
        collection = SimpleNamespace(collection_id=101)
        bindings = FakeBindingRepository()
        service, uow = self._binding_service(FakeCollectionRepository(existing=collection), bindings)

        binding = await service.bind_agent(
            BindAgentCollectionCommand(domain_id=8, collection_key="assets", agent_id="42")
        )

        self.assertEqual(201, binding.binding_id)
        self.assertEqual("AGENT", bindings.added.consumer_type)
        self.assertEqual("42", bindings.added.consumer_id)
        self.assertEqual("ACTIVE", bindings.added.status)
        uow.commit.assert_awaited_once()

    async def test_binding_rejects_collection_outside_scope(self):
        service, uow = self._binding_service(FakeCollectionRepository(), FakeBindingRepository())

        with self.assertRaises(CollectionNotFoundError):
            await service.bind_agent(
                BindAgentCollectionCommand(domain_id=8, collection_key="assets", agent_id="42")
            )

        uow.commit.assert_not_awaited()

class KnowledgeCoreCollectionLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_status_change_is_explicit_and_scoped(self):
        collection = SimpleNamespace(status="ACTIVE", updated_by=None)
        repo = FakeCollectionRepository(existing=collection)
        uow = FakeUnitOfWork(repo)
        service = KnowledgeCoreCollectionService(app_id=112, uow_factory=lambda: uow)
        from knowledge_core.application.collections import ChangeCollectionStatusCommand
        result = await service.change_status(ChangeCollectionStatusCommand(
            domain_id=8, collection_key="assets", status="DISABLED", actor_id="tester",
        ))
        self.assertEqual("DISABLED", result.status)
        self.assertEqual("tester", result.updated_by)
        uow.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
