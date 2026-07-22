"""Local object-store adapter tests; no Oracle or V1 table dependency."""
import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path

from knowledge_core.adapters.local_object_store import LocalKnowledgeObjectStore, ObjectIntegrityError


class LocalObjectStoreTest(unittest.TestCase):
    def test_stages_verifies_and_publishes_immutable_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upload.pdf"
            source.write_bytes(b"abc")
            store = LocalKnowledgeObjectStore(root / "objects")
            digest = hashlib.sha256(b"abc").hexdigest()

            staged = asyncio.run(store.stage_file(
                receipt_id="r1", part_name="attachment_0", source_path=source,
                expected_sha256=digest, expected_size=3, detected_mime_type="application/pdf",
            ))
            published = asyncio.run(store.publish_staged(staged=staged, collection_id=2, document_id=3))

            self.assertFalse(Path(staged.uri).exists())
            self.assertEqual(Path(published.uri).read_bytes(), b"abc")
            self.assertIn("/kc/2/3/", published.uri)

    def test_rejects_digest_mismatch_and_removes_staging_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upload.txt"
            source.write_bytes(b"abc")
            store = LocalKnowledgeObjectStore(root / "objects")

            with self.assertRaises(ObjectIntegrityError):
                asyncio.run(store.stage_file(
                    receipt_id="r1", part_name="attachment_0", source_path=source,
                    expected_sha256="0" * 64, expected_size=3, detected_mime_type="text/plain",
                ))
            self.assertFalse((root / "objects" / "kc-staging" / "r1" / "attachment_0").exists())


if __name__ == "__main__":
    unittest.main()
