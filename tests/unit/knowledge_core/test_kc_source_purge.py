"""KC Revision 源对象清理测试。"""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from platform_core.identity import uuid7
from knowledge_core.application.source_purge import (
    KnowledgeCoreSourcePurgeService,
    RevisionSourcePurgeConflictError,
)


class _Uow:
    def __init__(self, *, revision, bundle, collection, members, version, references=0):
        self.collections = SimpleNamespace(get_by_id_scope=AsyncMock(return_value=collection))
        self.bundles = SimpleNamespace(get_by_id=AsyncMock(return_value=bundle))
        self.revisions = SimpleNamespace(get_by_id=AsyncMock(return_value=revision))
        self.members = SimpleNamespace(
            list_by_revision=AsyncMock(return_value=members),
            count_other_references=AsyncMock(return_value=references),
        )
        self.versions = SimpleNamespace(
            get_by_id=AsyncMock(return_value=version),
            count_other_storage_users=AsyncMock(return_value=0),
        )
        self.commit = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class SourcePurgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_purges_only_exclusive_source_object(self):
        collection_id = uuid7()
        revision_id, bundle_id, version_id = uuid7(), uuid7(), uuid7()
        revision = SimpleNamespace(
            bundle_revision_id=revision_id, bundle_id=bundle_id,
            collection_id=collection_id, approval_status="REJECTED",
            status="REJECTED",
        )
        bundle = SimpleNamespace(current_revision_id=None)
        collection = SimpleNamespace(collection_id=collection_id)
        member = SimpleNamespace(document_version_id=version_id)
        version = SimpleNamespace(
            document_version_id=version_id,
            storage_state="AVAILABLE",
            storage_uri="/objects/manual.pdf",
        )
        prepare = _Uow(
            revision=revision, bundle=bundle, collection=collection,
            members=[member], version=version,
        )
        finalize = _Uow(
            revision=revision, bundle=bundle, collection=collection,
            members=[member], version=version,
        )
        uows = iter((prepare, finalize))
        objects = SimpleNamespace(delete=AsyncMock())
        service = KnowledgeCoreSourcePurgeService(
            uow_factory=lambda: next(uows), object_store=objects,
        )

        result = await service.purge_revision_sources(
            domain_id=9,
            collection_id=collection_id,
            bundle_revision_id=revision_id,
        )

        objects.delete.assert_awaited_once_with("/objects/manual.pdf")
        self.assertEqual("PURGED", version.storage_state)
        self.assertEqual(1, result.purged_count)
        self.assertEqual(0, result.retained_count)

    async def test_retains_source_shared_by_another_revision(self):
        collection_id = uuid7()
        revision_id, version_id = uuid7(), uuid7()
        revision = SimpleNamespace(
            bundle_revision_id=revision_id, bundle_id=uuid7(),
            collection_id=collection_id, approval_status="REJECTED",
            status="REJECTED",
        )
        version = SimpleNamespace(
            document_version_id=version_id,
            storage_state="AVAILABLE", storage_uri="/objects/shared.pdf",
        )
        uow = _Uow(
            revision=revision,
            bundle=SimpleNamespace(current_revision_id=None),
            collection=SimpleNamespace(collection_id=collection_id),
            members=[SimpleNamespace(document_version_id=version_id)],
            version=version,
            references=1,
        )
        objects = SimpleNamespace(delete=AsyncMock())
        service = KnowledgeCoreSourcePurgeService(
            uow_factory=lambda: uow, object_store=objects,
        )

        result = await service.purge_revision_sources(
            domain_id=9,
            collection_id=collection_id,
            bundle_revision_id=revision_id,
        )

        objects.delete.assert_not_awaited()
        self.assertEqual("AVAILABLE", version.storage_state)
        self.assertEqual(1, result.retained_count)

    async def test_rejects_current_published_revision(self):
        collection_id = uuid7()
        revision_id = uuid7()
        revision = SimpleNamespace(
            bundle_revision_id=revision_id, bundle_id=uuid7(),
            collection_id=collection_id, approval_status="REJECTED",
            status="REJECTED",
        )
        uow = _Uow(
            revision=revision,
            bundle=SimpleNamespace(current_revision_id=revision_id),
            collection=SimpleNamespace(collection_id=collection_id),
            members=[], version=None,
        )
        service = KnowledgeCoreSourcePurgeService(
            uow_factory=lambda: uow,
            object_store=SimpleNamespace(delete=AsyncMock()),
        )

        with self.assertRaises(RevisionSourcePurgeConflictError):
            await service.purge_revision_sources(
                domain_id=9,
                collection_id=collection_id,
                bundle_revision_id=revision_id,
            )


if __name__ == "__main__":
    unittest.main()
