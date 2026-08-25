"""Knowledge Core Revision 与源文件预览范围测试。"""

from types import SimpleNamespace
import unittest

from knowledge_core.application.preview import (
    KnowledgeCorePreviewService,
    KnowledgePreviewNotFoundError,
)
from platform_core.identity import uuid7


class _Collections:
    def __init__(self, domain_id, collection_id):
        self.domain_id = domain_id
        self.collection_id = collection_id

    async def get_by_id_scope(self, *, domain_id, collection_id):
        if domain_id == self.domain_id and collection_id == self.collection_id:
            return SimpleNamespace(collection_id=collection_id)
        return None


class _Repository:
    def __init__(self, key, values):
        self._key = key
        self._values = values

    async def get_by_id(self, **kwargs):
        expected = kwargs[self._key]
        return next(
            (
                value
                for value in self._values
                if getattr(value, self._key) == expected
            ),
            None,
        )


class _Members:
    def __init__(self, values):
        self._values = values

    async def list_by_revision(self, *, bundle_revision_id):
        return [
            value
            for value in self._values
            if value.bundle_revision_id == bundle_revision_id
        ]

    async def get_by_version(
        self, *, bundle_revision_id, document_version_id
    ):
        return next(
            (
                value
                for value in self._values
                if value.bundle_revision_id == bundle_revision_id
                and value.document_version_id == document_version_id
            ),
            None,
        )


class _Uow:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None


class KnowledgePreviewServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.domain_id = 20
        self.collection_id = uuid7()
        self.bundle_id = uuid7()
        self.revision_id = uuid7()
        self.version_id = uuid7()
        self.document_id = uuid7()
        self.bundle = SimpleNamespace(
            bundle_id=self.bundle_id,
            collection_id=self.collection_id,
            current_revision_id=self.revision_id,
        )
        self.revision = SimpleNamespace(
            bundle_id=self.bundle_id,
            collection_id=self.collection_id,
            bundle_revision_id=self.revision_id,
            title="产品说明资料",
            revision_no=2,
            status="READY",
            approval_status="APPROVED",
            manifest_json={
                "source_id": "ASSET-100",
                "metadata": {
                    "metadata_schema": "km_asset/v1",
                    "asset_title": "产品说明资料",
                    "author_mail": "owner@example.com",
                    "solution_briefing": "用于说明结构化预览字段。",
                    "publish_date": "2026-08-01",
                    "last_update_time": "2026-08-20T12:30:00Z",
                    "asset_product": "OAC",
                },
            },
        )
        self.member = SimpleNamespace(
            bundle_revision_id=self.revision_id,
            collection_id=self.collection_id,
            document_id=self.document_id,
            document_version_id=self.version_id,
            external_document_id="manual.pdf",
            declared_name="产品说明.pdf",
            document_role="CONTENT",
            ordinal=1,
            member_status="READY",
            declared_mime_type="application/pdf",
        )
        self.version = SimpleNamespace(
            document_version_id=self.version_id,
            collection_id=self.collection_id,
            bundle_id=self.bundle_id,
            document_id=self.document_id,
            storage_state="AVAILABLE",
            storage_uri="/private/kc/manual.pdf",
            detected_mime_type="application/pdf",
            byte_size=4096,
        )

    def service(self):
        uow = _Uow()
        uow.collections = _Collections(self.domain_id, self.collection_id)
        uow.bundles = _Repository("bundle_id", [self.bundle])
        uow.revisions = _Repository(
            "bundle_revision_id", [self.revision]
        )
        uow.members = _Members([self.member])
        uow.versions = _Repository(
            "document_version_id", [self.version]
        )
        return KnowledgeCorePreviewService(uow_factory=lambda: uow)

    async def test_bundle_preview_returns_ordered_source_file_metadata(self):
        result = await self.service().get_bundle_revision(
            domain_id=self.domain_id,
            collection_id=self.collection_id,
            bundle_id=self.bundle_id,
            bundle_revision_id=self.revision_id,
        )
        self.assertEqual("产品说明资料", result.title)
        self.assertEqual(self.version_id, result.files[0].document_version_id)
        self.assertTrue(result.files[0].preview_available)
        self.assertTrue(result.is_current_revision)
        self.assertEqual("ASSET-100", result.asset_fields["asset_id"])
        self.assertEqual(
            "owner@example.com", result.asset_fields["author_email"]
        )
        self.assertEqual(
            "用于说明结构化预览字段。", result.asset_fields["briefing"]
        )
        self.assertEqual("2026-08-01", result.asset_fields["publish_time"])
        self.assertEqual(
            "2026-08-20T12:30:00Z",
            result.asset_fields["last_update_time"],
        )
        self.assertEqual("OAC", result.asset_fields["product"])

    async def test_non_km_manifest_does_not_expose_source_metadata(self):
        self.revision.manifest_json = {
            "source_id": "DOCUMENT-100",
            "metadata": {"author_mail": "private@example.com"},
        }

        result = await self.service().get_bundle_revision(
            domain_id=self.domain_id,
            collection_id=self.collection_id,
            bundle_id=self.bundle_id,
            bundle_revision_id=self.revision_id,
        )

        self.assertEqual({}, result.asset_fields)

    async def test_source_file_requires_exact_revision_membership(self):
        result = await self.service().get_source_file(
            domain_id=self.domain_id,
            collection_id=self.collection_id,
            bundle_id=self.bundle_id,
            bundle_revision_id=self.revision_id,
            document_version_id=self.version_id,
        )
        self.assertEqual("产品说明.pdf", result.file_name)
        self.assertEqual(4096, result.byte_size)

        self.version.bundle_id = uuid7()
        with self.assertRaises(KnowledgePreviewNotFoundError):
            await self.service().get_source_file(
                domain_id=self.domain_id,
                collection_id=self.collection_id,
                bundle_id=self.bundle_id,
                bundle_revision_id=self.revision_id,
                document_version_id=self.version_id,
            )

    async def test_cross_domain_preview_is_hidden_as_not_found(self):
        with self.assertRaises(KnowledgePreviewNotFoundError):
            await self.service().get_bundle_revision(
                domain_id=21,
                collection_id=self.collection_id,
                bundle_id=self.bundle_id,
                bundle_revision_id=self.revision_id,
            )


if __name__ == "__main__":
    unittest.main()
