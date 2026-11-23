"""运维知识模型配置缺失与平台模型路由回归。"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_documents import OperationsDocuments
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.contracts.knowledge import ManualUploadMetadata
from aiops_agent.composition.knowledge_profiles import KnowledgeFoundationConfig
from knowledge_algorithms.contracts.identity import fingerprint
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.projections.contracts import ProjectionProfile, RepresentationSpec
from platform_core.identity import uuid7


def dense_spec():
    return RepresentationSpec(str(uuid7()), "model-checkpoint", "processor-sha256",
        "plain-text", 2, "DENSE", "TEXT", "COSINE", "L2", frozenset({"TEXT_TO_TEXT"}))


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.spec = dense_spec()
        self.profile = ProjectionProfile("test/v1", 100, 1000, 100, 1, 1000, 30., (self.spec,))
        self.documents = OperationsDocuments(uow_factory=None, ingestion=None, projection=None,
            retrieval=None, parse_profile=None, pipeline=SimpleNamespace(profile=self.profile))

    def test_matching_models_keep_exact_profile_fingerprint(self):
        models = {"embedding": self.spec.model_ref, "parser_llm": str(uuid7())}
        self.assertEqual(self.documents.configuration(models), {
            "models": models, "profile_ref": fingerprint(encode(self.profile)),
        })

    def test_missing_or_different_space_has_actionable_business_error(self):
        for models in ({}, {"embedding": str(uuid7())},
                       {"embedding": self.spec.model_ref, "visual_embedding": str(uuid7())}):
            with self.subTest(models=models), self.assertRaises(AIOpsApplicationError) as caught:
                self.documents.configuration(models)
            self.assertEqual(caught.exception.code, "AIOPS_KNOWLEDGE_PROFILE_MISMATCH")
            self.assertEqual(caught.exception.status_code, 409)
            self.assertIn("服务端表示空间", caught.exception.message)


class ManualProcessingPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_profile_without_vectors_keeps_capabilities_honest(self):
        parse, projection, profiles = KnowledgeFoundationConfig().profiles()
        documents = OperationsDocuments(uow_factory=None, ingestion=None, projection=None,
            retrieval=SimpleNamespace(profiles=profiles), parse_profile=parse,
            pipeline=SimpleNamespace(profile=projection))
        capabilities = documents.capability_view({})
        self.assertEqual({
            "native_parse": "PENDING_VERIFICATION", "ocr": "NOT_ENABLED",
            "visual_enrichment": "NOT_ENABLED", "lexical": "PENDING_VERIFICATION",
            "semantic_directions": [],
        }, capabilities)
        self.assertEqual("AVAILABLE", documents.capability_view(
            {"verified_parse_profiles": {parse.profile_hash}})["native_parse"])
        self.assertEqual(parse.profile_hash, documents.processing_plan()["parse"])

    async def test_ready_generation_from_incompatible_profile_is_not_reused(self):
        documents = OperationsDocuments.__new__(OperationsDocuments)
        plan = {"projection": "current"}
        documents.processing_plan = lambda **_: plan
        documents.inspect = AsyncMock(return_value={
            "status": "STAGED", "processing_status": "PARSED", "generation_id": str(uuid7()),
            "projection_profile_hash": "different",
        })
        with self.assertRaises(AIOpsApplicationError) as caught:
            await documents.process(None, uuid7(), expected_plan=plan)
        self.assertEqual("AIOPS_KNOWLEDGE_PROCESSING_PLAN_MISMATCH", caught.exception.code)

    async def test_upload_registers_without_resolving_models_or_index_profile(self):
        from aiops_agent.entities import OperationsKnowledgeAssetEntity
        oracle = OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_")
        domain = 42 if oracle else uuid7()
        version_id = uuid7()
        plan = {"projection": "registered-profile"}
        documents = SimpleNamespace(access=lambda _: SimpleNamespace(domain_id=domain, tenant_id=uuid7()),
            stage=AsyncMock(return_value={"source":SimpleNamespace(sha256="a"*64), "document_version_id":version_id}),
            register=AsyncMock(return_value={"document_id":uuid7(), "document_version_id":version_id,
                "ingestion_job_id":uuid7()}),
            processing_plan=unittest.mock.Mock(return_value=plan), assert_processing_plan=unittest.mock.Mock(),
            retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None)))
        base = SimpleNamespace(status="ACTIVE", configuration_json={})
        repo = SimpleNamespace(get_knowledge_base=AsyncMock(return_value=base),
            find_upload_request=AsyncMock(return_value=None), add_asset_version=AsyncMock())
        uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock())
        @asynccontextmanager
        async def factory():
            yield uow
        service = OperationsKnowledgeService(uow_factory=factory, documents=documents)
        args = dict(domain_id=domain, knowledge_base_id=uuid7(),
            context=SimpleNamespace(request_id="configuration-test", asserted_user_id="tester"),
            title="运维资料", documents=[{"file_name":"test.txt", "media_type":"text/plain", "ordinal":1,
                "role":"PRIMARY", "required_flag":True,
                "body":b"test", "content_sha256":"a" * 64}],
            metadata=ManualUploadMetadata(display_name="运维资料"), idempotency_key="configuration-test")
        with patch("aiops_agent.application.operations_knowledge._asset_view", side_effect=lambda row: {"status": row.status}), patch(
                "aiops_agent.application.operations_knowledge._version_view", side_effect=lambda row: {"status": row.status}), patch(
                "aiops_agent.application.operations_knowledge._index_view", side_effect=lambda row: {"status": row.index_status}):
            result = await service._register_manual_documents(**args)
        self.assertEqual("REVIEW_REQUIRED", result["version"]["status"])
        self.assertEqual([{"status": "PENDING_APPROVAL"}], result["documents"])
        recorded = repo.add_asset_version.await_args.kwargs["sources"][-1]
        self.assertFalse((recorded.source_locator_json or {}).get("processing_plan"))
        self.assertTrue(documents.register.await_args.kwargs["awaiting_approval"])
        documents.assert_processing_plan.assert_not_called()
        uow.commit.assert_awaited_once()

    async def test_existing_upload_can_replay_after_configuration_is_removed(self):
        from aiops_agent.entities import OperationsKnowledgeAssetEntity
        oracle = OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_")
        domain = 42 if oracle else uuid7()
        metadata = ManualUploadMetadata(display_name="运维资料")
        files = [{"file_name":"test.txt", "media_type":"text/plain", "body":b"test",
                  "content_sha256":"a" * 64}]
        digest = fingerprint({"title":metadata.display_name, "metadata":metadata.model_dump(mode="json"),
            "asset_id":None, "files":[{key:value for key,value in files[0].items() if key != "body"}]})
        documents = SimpleNamespace(access=lambda _:SimpleNamespace(domain_id=domain, tenant_id=uuid7()),
            stage=AsyncMock(return_value={}), configuration=unittest.mock.Mock(),
            retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None)))
        base = SimpleNamespace(status="ACTIVE", configuration_json={})
        existing = (object(), object(), SimpleNamespace(content_hash=digest))
        repo = SimpleNamespace(get_knowledge_base=AsyncMock(return_value=base),
            find_upload_request=AsyncMock(return_value=existing))
        uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock())
        @asynccontextmanager
        async def factory():
            yield uow
        kwargs = {} if oracle else {"model_catalog":None}
        service = OperationsKnowledgeService(uow_factory=factory, documents=documents, **kwargs)
        with patch("aiops_agent.application.operations_knowledge._asset_view", return_value={"asset_id":"existing"}), patch(
                "aiops_agent.application.operations_knowledge._version_view", return_value={"asset_version_id":"existing"}):
            result = await service._register_manual_documents(domain_id=domain, knowledge_base_id=uuid7(),
                context=SimpleNamespace(request_id="replay", asserted_user_id="tester", principal_id=uuid7()),
                title=metadata.display_name, documents=files, metadata=metadata, idempotency_key="replay")
        self.assertTrue(result["replayed"])
        documents.configuration.assert_not_called()
        uow.commit.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
