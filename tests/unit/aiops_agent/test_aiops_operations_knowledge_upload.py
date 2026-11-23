"""独立文件上传及单文件持久化合同回归，不连接数据库。"""

import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiops_agent.api import operations_knowledge as api
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.operations_knowledge import OperationsKnowledgeService
from aiops_agent.contracts.knowledge import ManualUploadMetadata
from aiops_agent.entities import OperationsKnowledgeAssetEntity, OperationsKnowledgeIndexEntity
from platform_core.identity import uuid7


ORACLE = OperationsKnowledgeAssetEntity.__tablename__.startswith("KBOT_")


def upload_documents():
    return [
        {
            "client_file_id": "manual",
            "file_name": "manual.md",
            "display_name": "manual.md",
            "media_type": "text/markdown",
            "body": b"# Manual",
            "ordinal": 0,
            "role": "CONTENT",
            "required_flag": True,
        },
        {
            "client_file_id": "appendix",
            "file_name": "appendix.txt",
            "display_name": "appendix.txt",
            "media_type": "text/plain",
            "body": b"Appendix",
            "ordinal": 1,
            "role": "SUPPLEMENT",
            "required_flag": False,
        },
    ]


class OperationsKnowledgeMultipartTest(unittest.TestCase):
    def setUp(self):
        self.service = SimpleNamespace(ingest_upload=AsyncMock(return_value={
            "grouping_mode": "EACH_FILE", "items": [],
        }))
        self.context = SimpleNamespace(request_id="multi-file-upload-test")
        self.domain_id = 42 if ORACLE else uuid7()
        self.tenant_id = uuid7()
        scope = (self.domain_id, "tester") if ORACLE else (
            self.tenant_id, self.domain_id, "tester",
        )
        scope_patch = patch.object(api, "_scope", return_value=scope)
        scope_patch.start()
        self.addCleanup(scope_patch.stop)
        app = FastAPI()
        app.state.operations_knowledge_service = self.service
        app.dependency_overrides[api.get_aiops_auth_context] = lambda: self.context
        app.include_router(api.router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.documents = upload_documents()
        self.declarations = [
            {
                "part_name": f"file-{index}",
                "client_file_id": doc["client_file_id"],
                "display_name": doc["display_name"],
                "byte_size": len(doc["body"]),
                "content_sha256": hashlib.sha256(doc["body"]).hexdigest(),
                "role": doc["role"],
                "required_flag": doc["required_flag"],
            }
            for index, doc in enumerate(self.documents)
        ]
        self.fields = {
            "knowledge_base_id": str(uuid7()),
            "grouping_mode": "EACH_FILE",
            "metadata": json.dumps({"display_name": "  数据库运维资料  ", "security_level": 2}),
            "files": json.dumps(self.declarations),
        }

    def post_upload(self):
        return self.client.post(
            "/internal/v1/aiops/operations-knowledge/uploads:ingest",
            data=self.fields,
            files=[
                (f"file-{index}", (doc["file_name"], doc["body"], doc["media_type"]))
                for index, doc in enumerate(self.documents)
            ],
            headers={"Idempotency-Key": "multi-file-upload-test"},
        )

    def test_multi_file_request_forwards_independent_files(self):
        response = self.post_upload()
        self.assertEqual(202, response.status_code, response.text)
        forwarded = self.service.ingest_upload.await_args.kwargs
        self.assertEqual("EACH_FILE", forwarded["grouping_mode"])
        self.assertEqual("数据库运维资料", forwarded["metadata"].display_name)
        self.assertEqual(self.domain_id, forwarded["domain_id"])
        self.assertEqual([("manual.md", b"# Manual"), ("appendix.txt", b"Appendix")],
                         [(doc["file_name"], doc["body"]) for doc in forwarded["documents"]])
        self.assertTrue(all(not {"ordinal", "role", "required_flag"} & doc.keys()
                            for doc in forwarded["documents"]))

    def test_each_file_mode_remains_supported(self):
        self.fields["grouping_mode"] = "EACH_FILE"
        response = self.post_upload()
        self.assertEqual(202, response.status_code, response.text)
        self.assertEqual("EACH_FILE", self.service.ingest_upload.await_args.kwargs["grouping_mode"])

    def test_obsolete_and_unknown_grouping_modes_are_rejected(self):
        for mode in ("SINGLE_ASSET", "SINGLE_BUNDLE", "UNKNOWN"):
            with self.subTest(mode=mode):
                self.fields["grouping_mode"] = mode
                response = self.post_upload()
                self.assertEqual(422, response.status_code, response.text)
                self.assertEqual("AIOPS_KNOWLEDGE_UPLOAD_INVALID", response.json()["detail"]["code"])
        self.service.ingest_upload.assert_not_awaited()

    def test_upload_requires_metadata_title(self):
        self.fields["metadata"] = json.dumps({"security_level": 2})
        response = self.post_upload()
        self.assertEqual(422, response.status_code, response.text)
        self.service.ingest_upload.assert_not_awaited()

    def test_file_digest_is_still_verified(self):
        self.declarations[1]["content_sha256"] = "0" * 64
        self.fields["files"] = json.dumps(self.declarations)
        response = self.post_upload()
        self.assertEqual(422, response.status_code, response.text)
        self.assertIn("摘要不一致", response.json()["detail"]["message"])
        self.service.ingest_upload.assert_not_awaited()

    def test_blank_metadata_title_is_rejected(self):
        for title in ("", "   "):
            with self.subTest(title=title):
                self.fields["metadata"] = json.dumps({"display_name": title})
                response = self.post_upload()
                self.assertEqual(422, response.status_code, response.text)
        self.service.ingest_upload.assert_not_awaited()


class OperationsKnowledgeGroupingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = OperationsKnowledgeService(
            uow_factory=lambda: None,
            documents=SimpleNamespace(
                retrieval=SimpleNamespace(authorization=SimpleNamespace(service=None)),
            ),
        )
        self.arguments = {
            "domain_id": 42 if ORACLE else uuid7(),
            "knowledge_base_id": uuid7(),
            "context": SimpleNamespace(request_id="multi-file-upload-test"),
            "grouping_mode": "EACH_FILE",
            "documents": upload_documents(),
            "metadata": ManualUploadMetadata(display_name="数据库运维资料"),
            "idempotency_key": "multi-file-upload-test",
        }
        if not ORACLE:
            self.arguments["tenant_id"] = uuid7()
        for document in self.arguments["documents"]:
            document["content_sha256"] = hashlib.sha256(document["body"]).hexdigest()

    def test_one_file_mapping_per_asset_version_in_schema(self):
        table = OperationsKnowledgeIndexEntity.__table__
        self.assertEqual(["asset_version_id"], [column.key for column in table.primary_key])
        self.assertFalse(table.c.document_version_id.nullable)

    async def test_each_file_creates_independent_assets(self):
        asset_ids = [str(uuid7()), str(uuid7())]
        self.service.ingest_manual = AsyncMock(side_effect=[
            {"asset": {"asset_id": asset_id}, "version": {"asset_version_id": str(uuid7())}}
            for asset_id in asset_ids
        ])
        self.arguments["grouping_mode"] = "EACH_FILE"
        receipt = await self.service.ingest_upload(**self.arguments)
        self.assertEqual(asset_ids, [item["asset_id"] for item in receipt["items"]])
        calls = [call.kwargs for call in self.service.ingest_manual.await_args_list]
        self.assertEqual(["manual.md", "appendix.txt"], [call["metadata"].display_name for call in calls])
        self.assertEqual(
            ["multi-file-upload-test:manual", "multi-file-upload-test:appendix"],
            [call["idempotency_key"] for call in calls],
        )

    async def test_failed_file_does_not_block_independent_file(self):
        self.service.ingest_manual = AsyncMock(side_effect=[
            AIOpsApplicationError(code="AIOPS_KNOWLEDGE_UPLOAD_INVALID", message="登记失败", status_code=422),
            {"asset": {"asset_id": str(uuid7())}, "version": {"asset_version_id": str(uuid7())}},
        ])
        receipt = await self.service.ingest_upload(**self.arguments)
        self.assertEqual(["REJECTED", "ACCEPTED"], [item["status"] for item in receipt["items"]])

    async def test_application_rejects_group_creation(self):
        self.service.ingest_manual = AsyncMock()
        self.arguments["grouping_mode"] = "SINGLE_ASSET"
        with self.assertRaises(AIOpsApplicationError) as raised:
            await self.service.ingest_upload(**self.arguments)
        self.assertEqual(422, raised.exception.status_code)
        self.service.ingest_manual.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
