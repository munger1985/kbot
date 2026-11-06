"""正式 App 业务闭环的合成回归；仅 SQLite 内存与受控解析替身。"""

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from uuid import UUID

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [
    str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_ingestion"), str(ROOT / "tests/unit/knowledge_algorithms")]
import test_app_knowledge_retrieval as support
from knowledge_algorithm_fixtures import CharacterTokenizer, parse_profile, projection_profile
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.projections.builder import ProjectionBuilder
from platform_core.contracts.knowledge_app import KnowledgeSearchRequest
from knowledge_retrieval_app.entities.knowledge_library import KnowledgeLibraryEntity
from knowledge_retrieval_app.repositories.knowledge_library import KnowledgeLibraryRepository
from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService, LibraryAuthorization


class SyntheticSession(support.AsyncSyntheticSession):
    async def scalar(self, *args, **kwargs):
        return self.session.scalar(*args, **kwargs)


class LibraryBusinessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await support.AppRetrievalTests.asyncSetUp(self)
        KnowledgeLibraryEntity.__table__.create(self.engine)
        self.fixture.sessions = lambda: SyntheticSession(self.engine)
        previous = self.uow_factory
        class BusinessUow:
            async def __aenter__(inner):
                inner.delegate = previous()
                value = await inner.delegate.__aenter__()
                value.library = lambda scope: KnowledgeLibraryRepository(value.session, scope)
                from knowledge_retrieval_app.repositories.knowledge_queue import KnowledgeJobQueueRepository
                value.knowledge_queue = KnowledgeJobQueueRepository(value.session, lambda: None)
                return value
            async def __aexit__(inner, *args):
                return await inner.delegate.__aexit__(*args)
        self.uow_factory = BusinessUow
        self.fixture.service.uow_factory = self.uow_factory
        self.service.uow_factory = self.uow_factory
        self.service.authorization = LibraryAuthorization(self.uow_factory)
        self.service.profiles = {"default": self.service.profiles["baseline"]}
        self.business = KnowledgeLibraryService(uow_factory=self.uow_factory,
            ingestion=self.fixture.service, retrieval=self.service,
            parse_profile=parse_profile(), configurations={"default": object()})
        created = await self.business.create(self.fixture.scope, display_name="合成资料库",
                                            description=None, profile_ref="default")
        self.library_id = UUID(created["library_id"])

    async def asyncTearDown(self):
        await support.AppRetrievalTests.asyncTearDown(self)

    async def upload(self, key="upload-key", raw="论文图文与知识检索。".encode()):
        return await self.business.upload(self.fixture.scope, self.library_id,
            name="synthetic.txt", mime_type="text/plain", raw=raw, idempotency_key=key)

    async def test_upload_idempotency_conflict_and_private_acl(self):
        first = await self.upload()
        self.assertEqual(first["version_id"], (await self.upload())["version_id"])
        with self.assertRaises(AlgorithmError):
            await self.upload(raw=b"different")
        with self.assertRaises(AlgorithmError):
            await self.business.files(replace(self.fixture.scope, subject_ref="other"), self.library_id,
                                      page=1, page_size=20)

    async def test_review_parse_publish_search_withdraw(self):
        value = await self.upload()
        version_id = UUID(value["version_id"])
        self.fixture.job = UUID(value["job_id"])
        self.fixture.version = version_id
        # 未审核/未索引不能发布；尚无发布资料时检索如实返回零结果。
        with self.assertRaises(AlgorithmError):
            await self.business.publish(self.fixture.scope, self.library_id, version_id)
        empty = await self.business.search(self.fixture.scope, self.library_id,
                                          KnowledgeSearchRequest(text="知识检索"))
        self.assertFalse(empty["primary_hits"])
        await self.fixture.parse()
        async def original(content):
            return content
        document = await self.fixture.projections.allowed_document(self.fixture.scope, self.fixture.job,
            policy=original, policy_revision="synthetic/v1")
        await self.fixture.projections.prepare(self.fixture.scope, self.fixture.job, document=document,
            builder=ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None), profile=projection_profile())
        await self.business.review(self.fixture.scope, self.library_id, version_id,
                                   decision="APPROVE", comment="合成审核")
        self.assertEqual((await self.business.publish(self.fixture.scope, self.library_id, version_id))["status"], "PUBLISHED")
        result = await self.business.search(self.fixture.scope, self.library_id,
                                           KnowledgeSearchRequest(text="知识检索", requested_count=2))
        self.assertTrue(result["primary_hits"])
        self.assertIn("知识检索", result["items"][0]["text"])
        self.assertNotIn("content_handle", repr(result))
        await self.business.withdraw(self.fixture.scope, self.library_id, version_id)
        with self.assertRaises(AlgorithmError):
            await self.service.receipt(self.fixture.scope, result["retrieval_ref"])

    async def test_grants_revision_invalidates_scope_and_prevents_last_manager_removal(self):
        scope = await self.service.create_scope(self.fixture.scope, resource_ref="library:" + str(self.library_id))
        with self.assertRaises(AlgorithmError):
            await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                       changes={"grants": {"reader": ["READ"]}})
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                   changes={"display_name": "更新后的合成资料库"})
        with self.assertRaises(AlgorithmError):
            await self.service.validate(self.fixture.scope, scope)

    async def test_unpublished_visibility_and_stream_reauthorization(self):
        value = await self.upload()
        version_id = UUID(value["version_id"])
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
            changes={"grants": {self.fixture.scope.subject_ref: ["READ", "UPLOAD", "REVIEW", "MANAGE", "PUBLISH", "WITHDRAW"],
                               "reader": ["READ"]}})
        reader = replace(self.fixture.scope, subject_ref="reader")
        self.assertFalse((await self.business.files(reader, self.library_id, page=1, page_size=20))["items"])
        with self.assertRaises(AlgorithmError):
            await self.business.source(reader, self.library_id, version_id)
        files = await self.business.files(self.fixture.scope, self.library_id, page=1, page_size=20)
        version = files["items"][0]["versions"][value["version_id"]]
        self.assertEqual(version["processing_status"], "RECEIVED")
        self.assertFalse(version["generation_ready"])
        _, stream = await self.business.source(self.fixture.scope, self.library_id, version_id)
        await self.business.update(self.fixture.scope, self.library_id, expected_revision=2,
            changes={"status": "DISABLED"})
        with self.assertRaises(AlgorithmError):
            _ = b"".join([chunk async for chunk in stream])

    async def test_formal_queue_resumes_projection_on_another_worker(self):
        from knowledge_retrieval_app.application.knowledge_pipeline import KnowledgePipelineWorker
        from knowledge_retrieval_app.application.knowledge_worker import KnowledgeBusinessWorker
        value = await self.upload()
        self.fixture.job, self.fixture.version = UUID(value["job_id"]), UUID(value["version_id"])
        # 解析由 fixture 原 Worker 完成，索引由另一独立 Worker 接管。
        await self.fixture.parse()
        async def original(content):
            return content
        pipeline = KnowledgePipelineWorker(ingestion=self.fixture.service,
            projection=self.fixture.projections,
            builder_factory=lambda scope, document: ProjectionBuilder(self.fixture.binding, CharacterTokenizer(), None),
            profile=projection_profile(), allowed_policy=original, policy_revision="synthetic/v1",
            worker_id="another-worker")
        worker = KnowledgeBusinessWorker(uow_factory=self.uow_factory, pipeline=pipeline,
                                         worker_id="another-worker", poll_seconds=.1)
        self.assertTrue(await worker.process_once())
        rows = await self.business.files(self.fixture.scope, self.library_id, page=1, page_size=20)
        version = rows["items"][0]["versions"][value["version_id"]]
        self.assertEqual(version["processing_status"], "READY")
        self.assertTrue(version["generation_ready"])
        self.assertFalse(await worker.process_once())
        self.assertEqual(version["status"], "PARSED")

    async def test_invalid_partial_updates_and_grants(self):
        from pydantic import ValidationError
        from platform_core.contracts.knowledge_app import KnowledgeLibraryUpdate
        for field in ("display_name", "status", "grants"):
            with self.assertRaises(ValidationError):
                KnowledgeLibraryUpdate.model_validate({"expected_revision": 1, field: None})
        for grants in ({self.fixture.scope.subject_ref: ["MANAGE"]},
                       {self.fixture.scope.subject_ref: ["READ", "MANAGE"], "reader": ["READ", "READ"]}):
            with self.assertRaises(AlgorithmError):
                await self.business.update(self.fixture.scope, self.library_id, expected_revision=1,
                                           changes={"grants": grants})

    async def test_public_query_rejects_scope_models_and_unregistered_policy(self):
        from pydantic import ValidationError
        for key in ("scope", "profile_ref", "model_id", "owner_binding_ref"):
            with self.assertRaises(ValidationError):
                KnowledgeSearchRequest.model_validate({"text": "test", key: "injected"})
        with self.assertRaises(ValidationError):
            KnowledgeSearchRequest(text="x", mode="FILTERED_ORDER")
        with self.assertRaises(AlgorithmError):
            await self.business.create(self.fixture.scope, display_name="非法配置",
                                       description=None, profile_ref="unregistered")


if __name__ == "__main__":
    unittest.main()
