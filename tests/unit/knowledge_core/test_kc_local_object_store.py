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
            self.assertFalse(
                (root / "objects" / "kc-staging" / "r1").exists()
            )
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
            self.assertFalse(
                (root / "objects" / "kc-staging" / "r1").exists()
            )

    def test_delete_removes_only_empty_managed_parents(self):
        """删除对象时保留同目录下仍在使用的其他对象。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_root = root / "objects"
            first = object_root / "kc" / "collection-1" / "doc-1" / "hash-1"
            second = object_root / "kc" / "collection-1" / "doc-2" / "hash-2"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            store = LocalKnowledgeObjectStore(object_root)

            asyncio.run(store.delete(str(first)))

            self.assertFalse(first.exists())
            self.assertFalse(first.parent.exists())
            self.assertTrue(second.exists())
            self.assertTrue(second.parent.exists())
            self.assertTrue((object_root / "kc").exists())

    def test_delete_rejects_path_outside_object_root(self):
        """清理逻辑不能越过配置的对象存储根目录。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.txt"
            outside.write_bytes(b"keep")
            store = LocalKnowledgeObjectStore(root / "objects")

            with self.assertRaisesRegex(ValueError, "不属于"):
                asyncio.run(store.delete(str(outside)))

            self.assertTrue(outside.exists())

    def test_stream_supports_range_without_buffering_whole_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_root = root / "objects"
            stored = object_root / "kc" / "collection-1" / "large.bin"
            stored.parent.mkdir(parents=True)
            stored.write_bytes(b"0123456789")
            store = LocalKnowledgeObjectStore(object_root)

            async def read() -> bytes:
                self.assertEqual(10, await store.size(str(stored)))
                return b"".join(
                    [
                        chunk
                        async for chunk in store.stream(
                            str(stored),
                            offset=3,
                            length=4,
                            chunk_size=2,
                        )
                    ]
                )

            self.assertEqual(b"3456", asyncio.run(read()))


if __name__ == "__main__":
    unittest.main()
