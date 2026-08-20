import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_core.application.status import KnowledgeCoreStatusService
from knowledge_core.domain.revision_status import reduce_revision_status
from platform_core.identity import uuid7


def member(role, status):
    return SimpleNamespace(document_role=role, member_status=status)


class RevisionStatusTest(unittest.TestCase):
    def test_ready_requires_manifest_and_all_members_ready(self):
        self.assertEqual("READY", reduce_revision_status([member("MANIFEST", "READY"), member("ATTACHMENT", "READY")]))

    def test_partial_keeps_searchable_manifest_with_failed_attachment(self):
        self.assertEqual("PARTIAL", reduce_revision_status([member("MANIFEST", "READY"), member("ATTACHMENT", "SOURCE_UNAVAILABLE")]))

    def test_manifest_failure_fails_revision(self):
        self.assertEqual("FAILED", reduce_revision_status([member("MANIFEST", "FAILED"), member("ATTACHMENT", "READY")]))

    def test_pending_member_keeps_revision_processing(self):
        self.assertEqual("PROCESSING", reduce_revision_status([member("MANIFEST", "READY"), member("ATTACHMENT", "PARSING")]))

    def test_manifestless_upload_treats_documents_as_peers(self):
        self.assertEqual(
            "READY",
            reduce_revision_status(
                [
                    member("CONTENT", "READY"),
                    member("ATTACHMENT", "READY"),
                ]
            ),
        )
        self.assertEqual(
            "PARTIAL",
            reduce_revision_status(
                [
                    member("CONTENT", "READY"),
                    member("ATTACHMENT", "FAILED"),
                ]
            ),
        )
        self.assertEqual(
            "FAILED",
            reduce_revision_status([member("CONTENT", "FAILED")]),
        )

    def test_manifestless_upload_waits_for_all_documents(self):
        self.assertEqual(
            "PROCESSING",
            reduce_revision_status(
                [
                    member("CONTENT", "READY"),
                    member("ATTACHMENT", "INDEXING"),
                ]
            ),
        )


class DiscoveryReindexStatusTest(unittest.IsolatedAsyncioTestCase):
    async def test_succeeded_index_marks_operation_succeeded(self):
        collection_id = uuid7()
        bundle_id = uuid7()
        revision_id = uuid7()
        generation = uuid7()
        jobs = [
            SimpleNamespace(
                ingestion_job_id=uuid7(),
                job_type="PROFILE",
                job_status="SUCCEEDED",
                payload_json={"reindex_generation": str(generation)},
                attempt_count=1,
                failure_code=None,
                failure_message=None,
                started_at=None,
                completed_at=None,
            ),
            SimpleNamespace(
                ingestion_job_id=uuid7(),
                job_type="INDEX",
                job_status="SUCCEEDED",
                payload_json={"reindex_generation": str(generation)},
                attempt_count=1,
                failure_code=None,
                failure_message=None,
                started_at=None,
                completed_at=None,
            ),
        ]
        uow = SimpleNamespace(
            collections=SimpleNamespace(
                get_by_id_scope=AsyncMock(return_value=SimpleNamespace())
            ),
            bundles=SimpleNamespace(
                get_by_id=AsyncMock(return_value=SimpleNamespace(
                    bundle_id=bundle_id,
                    collection_id=collection_id,
                ))
            ),
            revisions=SimpleNamespace(
                get_by_id=AsyncMock(return_value=SimpleNamespace(
                    bundle_id=bundle_id,
                ))
            ),
            jobs=SimpleNamespace(
                list_by_revisions=AsyncMock(return_value=jobs)
            ),
        )

        class Uow:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, *_):
                return None

        result = await KnowledgeCoreStatusService(
            uow_factory=Uow
        ).get_discovery_reindex_operation(
            domain_id=43,
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            generation=generation,
        )

        self.assertEqual("SUCCEEDED", result.status)
        self.assertEqual(["PROFILE", "INDEX"], [item["job_type"] for item in result.jobs])


if __name__ == "__main__":
    unittest.main()
