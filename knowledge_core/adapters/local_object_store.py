"""Local immutable object store for development and integration tests.

Production deployment can replace this adapter with OCI Object Storage without
changing an intake use case.  Objects first live under a private staging
prefix, then move to an immutable content-addressed published key.
"""
import asyncio
import hashlib
import shutil
from pathlib import Path

from knowledge_core.ports.object_store import StoredObject


class ObjectIntegrityError(ValueError):
    """A stream differs from the declared digest or byte size."""


class ObjectAlreadyPublishedError(RuntimeError):
    """A content-addressed target exists with incompatible content."""


class LocalKnowledgeObjectStore:
    def __init__(self, root: Path):
        self._root = root.resolve()

    async def stage_file(
        self, *, receipt_id: str, part_name: str, source_path: Path, expected_sha256: str,
        expected_size: int, detected_mime_type: str,
    ) -> StoredObject:
        return await asyncio.to_thread(
            self._stage_file_sync, receipt_id, part_name, source_path, expected_sha256,
            expected_size, detected_mime_type,
        )

    def _stage_file_sync(
        self, receipt_id: str, part_name: str, source_path: Path, expected_sha256: str,
        expected_size: int, detected_mime_type: str,
    ) -> StoredObject:
        target = self._root / "kc-staging" / receipt_id / part_name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        byte_size = 0
        with source_path.open("rb") as reader, target.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                byte_size += len(chunk)
                writer.write(chunk)
        actual_hash = digest.hexdigest()
        if actual_hash.lower() != expected_sha256.lower() or byte_size != expected_size:
            target.unlink(missing_ok=True)
            raise ObjectIntegrityError("staged object does not match declared hash or size")
        return StoredObject(str(target), byte_size, actual_hash, detected_mime_type)

    async def publish_staged(
        self, *, staged: StoredObject, collection_id: int, document_id: int,
    ) -> StoredObject:
        return await asyncio.to_thread(self._publish_staged_sync, staged, collection_id, document_id)

    def _publish_staged_sync(self, staged: StoredObject, collection_id: int, document_id: int) -> StoredObject:
        source = Path(staged.uri)
        target = self._root / "kc" / str(collection_id) / str(document_id) / staged.content_sha256.lower()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != staged.byte_size:
                raise ObjectAlreadyPublishedError("immutable target exists with a different size")
            source.unlink(missing_ok=True)
        else:
            shutil.move(str(source), str(target))
        return StoredObject(str(target), staged.byte_size, staged.content_sha256.lower(), staged.detected_mime_type)

    async def delete(self, uri: str) -> None:
        await asyncio.to_thread(Path(uri).unlink, missing_ok=True)
