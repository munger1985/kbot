"""运维知识模型配置缺失与平台模型路由回归。"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_documents import OperationsDocuments
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.contracts.knowledge import ManualUploadMetadata
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


class MissingModelsTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_models_never_raise_key_error_or_register_documents(self):
        from aiops_agent.entities import OperationsKnowledgeAssetEntity
        oracle = OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_")
        domain = 42 if oracle else uuid7()
        documents = SimpleNamespace(access=lambda _: SimpleNamespace(domain_id=domain, tenant_id=uuid7()),
            stage=AsyncMock(return_value={}), configuration=unittest.mock.Mock(),
            retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None)))
        base = SimpleNamespace(status="ACTIVE", configuration_json={})
        repo = SimpleNamespace(get_knowledge_base=AsyncMock(return_value=base),
            find_upload_request=AsyncMock(return_value=None))
        uow = SimpleNamespace(operations_knowledge=repo, commit=AsyncMock())
        @asynccontextmanager
        async def factory():
            yield uow
        service = OperationsKnowledgeService(uow_factory=factory, documents=documents)
        args = dict(domain_id=domain, knowledge_base_id=uuid7(),
            context=SimpleNamespace(request_id="configuration-test", asserted_user_id="tester"),
            title="运维资料", documents=[{"file_name":"test.txt", "media_type":"text/plain",
                "body":b"test", "content_sha256":"a" * 64}],
            metadata=ManualUploadMetadata(display_name="运维资料"), idempotency_key="configuration-test")
        if not oracle:
            args["tenant_id"] = uuid7()
        for configuration in ({}, None, {"models": None}, {"models": {}},
                              {"models": {"embedding": str(uuid7())}}):
            base.configuration_json = configuration
            with self.subTest(configuration=configuration), self.assertRaises(AIOpsApplicationError) as caught:
                await service._register_manual_documents(**args)
            self.assertEqual(caught.exception.code, "AIOPS_KNOWLEDGE_COLLECTION_MODELS_REQUIRED")
            self.assertEqual(caught.exception.status_code, 409)
        uow.commit.assert_not_awaited()
        documents.stage.assert_not_awaited()
        documents.configuration.assert_not_called()

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
