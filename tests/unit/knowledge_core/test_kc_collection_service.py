"""Unit tests for Collection root creation without an Oracle dependency."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from platform_core.identity import uuid7

from knowledge_core.application.collections import (
    BindAgentCollectionCommand,
    CollectionNotFoundError,
    CreateCollectionCommand,
    KnowledgeCoreBindingService,
    KnowledgeCoreCollectionService,
    UpdateCollectionModelsCommand,
)

COLLECTION_ID = uuid7()
AGENT_ID = uuid7()


class FakeCollectionRepository:
    def __init__(self, existing=None):
        self._existing = existing
        self.added = None

    async def get_by_id_scope(self, **kwargs):
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

    async def has_active_binding(self, **kwargs):
        del kwargs
        return False


class FakeJobRepository:
    def __init__(self, existing=None):
        self.existing = existing
        self.added = None

    async def get_by_idempotency_key(self, **kwargs):
        del kwargs
        return self.existing

    async def add(self, job):
        self.added = job
        return job


class FakeUnitOfWork:
    def __init__(self, repository, binding_repository=None):
        self.collections = repository
        self.bindings = binding_repository
        self.session = SimpleNamespace(flush=AsyncMock())
        self.commit = AsyncMock()
        self.jobs = None
        self.flush = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class KnowledgeCoreCollectionServiceTest(unittest.IsolatedAsyncioTestCase):
    def _service(self, repository):
        uow = FakeUnitOfWork(repository)
        service = KnowledgeCoreCollectionService(uow_factory=lambda: uow)
        return service, uow

    def _binding_service(self, collection_repository, binding_repository):
        uow = FakeUnitOfWork(collection_repository, binding_repository)
        service = KnowledgeCoreBindingService(uow_factory=lambda: uow)
        return service, uow

    async def test_creates_collection_with_server_injected_scope(self):
        repository = FakeCollectionRepository()
        service, uow = self._service(repository)
        embedding_model_id = uuid7()

        collection = await service.create(
            CreateCollectionCommand(
                domain_id=8,
                display_name="Asset Knowledge",
                models={
                    "parser_llm": uuid7(),
                    "parser_vlm": uuid7(),
                    "retrieval_llm": uuid7(),
                    "embedding": embedding_model_id,
                },
                metadata={"source": "km"},
                actor_id="user:7",
            )
        )

        self.assertEqual(101, collection.collection_id)
        self.assertEqual(8, repository.added.domain_id)
        self.assertEqual(
            str(embedding_model_id),
            repository.added.models_json["embedding"],
        )
        self.assertEqual("ACTIVE", repository.added.status)
        uow.commit.assert_awaited_once()

    async def test_list_returns_session_independent_snapshots(self):
        collection = SimpleNamespace(
            collection_id=uuid7(),
            domain_id=8,
            display_name="Asset Knowledge",
            description=None,
            models_json={
                "parser_llm": str(uuid7()),
                "retrieval_llm": str(uuid7()),
                "embedding": str(uuid7()),
            },
            status="ACTIVE",
            default_security_level=1,
            metadata_json={"source": "test"},
            row_version=1,
            updated_at=datetime.now(timezone.utc),
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

        with self.assertRaisesRegex(ValueError, "embedding"):
            await service.create(CreateCollectionCommand(
                domain_id=8,
                display_name="Asset Knowledge",
                models={
                    "parser_llm": uuid7(),
                    "retrieval_llm": uuid7(),
                    "embedding": "not-a-uuid",
                },
            ))

        self.assertIsNone(repository.added)
        uow.commit.assert_not_awaited()

    async def test_updates_role_map_without_changing_embedding(self):
        original_embedding = uuid7()
        collection = SimpleNamespace(
            models_json={
                "parser_llm": str(uuid7()),
                "retrieval_llm": str(uuid7()),
                "embedding": str(original_embedding),
            },
            updated_by=None,
            row_version=1,
        )
        service, uow = self._service(
            FakeCollectionRepository(existing=collection)
        )
        parser_llm = uuid7()
        parser_vlm = uuid7()
        retrieval_llm = uuid7()

        updated = await service.update_models(
            UpdateCollectionModelsCommand(
                domain_id=8,
                collection_id=COLLECTION_ID,
                models={
                    "parser_llm": parser_llm,
                    "parser_vlm": parser_vlm,
                    "retrieval_llm": retrieval_llm,
                    "embedding": original_embedding,
                    "future_role": uuid7(),
                },
                actor_id="user:7",
            )
        )

        self.assertEqual(str(parser_llm), updated.models_json["parser_llm"])
        self.assertEqual(str(parser_vlm), updated.models_json["parser_vlm"])
        self.assertEqual(
            str(retrieval_llm), updated.models_json["retrieval_llm"]
        )
        self.assertEqual(
            str(original_embedding), updated.models_json["embedding"]
        )
        self.assertIn("future_role", updated.models_json)
        uow.commit.assert_awaited_once()

    async def test_binds_agent_to_collection_without_retrieval_weights(self):
        collection = SimpleNamespace(collection_id=COLLECTION_ID)
        bindings = FakeBindingRepository()
        service, uow = self._binding_service(FakeCollectionRepository(existing=collection), bindings)

        binding = await service.bind_agent(
            BindAgentCollectionCommand(
                domain_id=8,
                collection_id=COLLECTION_ID,
                agent_id=AGENT_ID,
            )
        )

        self.assertEqual(201, binding.binding_id)
        self.assertEqual("AGENT", bindings.added.consumer_type)
        self.assertEqual(AGENT_ID, bindings.added.consumer_id)
        self.assertEqual("ACTIVE", bindings.added.status)
        uow.commit.assert_awaited_once()

    async def test_binding_rejects_collection_outside_scope(self):
        service, uow = self._binding_service(FakeCollectionRepository(), FakeBindingRepository())

        with self.assertRaises(CollectionNotFoundError):
            await service.bind_agent(
                BindAgentCollectionCommand(
                    domain_id=8,
                    collection_id=COLLECTION_ID,
                    agent_id=AGENT_ID,
                )
            )

        uow.commit.assert_not_awaited()

class KnowledgeCoreCollectionLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_status_change_is_explicit_and_scoped(self):
        collection = SimpleNamespace(status="ACTIVE", updated_by=None)
        repo = FakeCollectionRepository(existing=collection)
        uow = FakeUnitOfWork(repo)
        service = KnowledgeCoreCollectionService(uow_factory=lambda: uow)
        from knowledge_core.application.collections import ChangeCollectionStatusCommand
        result = await service.change_status(ChangeCollectionStatusCommand(
            domain_id=8,
            collection_id=COLLECTION_ID,
            status="DISABLED",
            actor_id="tester",
        ))
        self.assertEqual("DISABLED", result.status)
        self.assertEqual("tester", result.updated_by)
        uow.commit.assert_awaited_once()

    async def test_repeated_delete_returns_existing_purge_job(self):
        job_id = uuid7()
        collection = SimpleNamespace(
            collection_id=COLLECTION_ID,
            status="DELETING",
            updated_by="tester",
        )
        existing = SimpleNamespace(
            ingestion_job_id=job_id,
            job_status="PENDING",
        )
        uow = FakeUnitOfWork(
            FakeCollectionRepository(existing=collection),
            FakeBindingRepository(),
        )
        uow.jobs = FakeJobRepository(existing=existing)
        service = KnowledgeCoreCollectionService(uow_factory=lambda: uow)

        returned = await service.request_delete(
            domain_id=8,
            collection_id=COLLECTION_ID,
            actor_id="tester",
        )

        self.assertEqual(job_id, returned)
        self.assertIsNone(uow.jobs.added)

    async def test_failed_purge_can_be_requeued_with_same_job(self):
        job_id = uuid7()
        collection = SimpleNamespace(
            collection_id=COLLECTION_ID,
            status="DELETION_FAILED",
            updated_by="tester",
        )
        existing = SimpleNamespace(
            ingestion_job_id=job_id,
            job_status="FAILED",
            attempt_count=3,
            available_at=None,
            failure_class="TRANSIENT",
            failure_code="OBJECT_DELETE_FAILED",
            failure_message="store down",
        )
        uow = FakeUnitOfWork(
            FakeCollectionRepository(existing=collection),
            FakeBindingRepository(),
        )
        uow.jobs = FakeJobRepository(existing=existing)
        service = KnowledgeCoreCollectionService(uow_factory=lambda: uow)

        returned = await service.request_delete(
            domain_id=8,
            collection_id=COLLECTION_ID,
            actor_id="tester",
        )

        self.assertEqual(job_id, returned)
        self.assertEqual("PENDING", existing.job_status)
        self.assertEqual(0, existing.attempt_count)
        self.assertEqual("DELETING", collection.status)
        uow.commit.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
