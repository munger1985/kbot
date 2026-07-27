import unittest
from types import SimpleNamespace

from knowledge_core.domain.revision_status import reduce_revision_status


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


if __name__ == "__main__":
    unittest.main()
