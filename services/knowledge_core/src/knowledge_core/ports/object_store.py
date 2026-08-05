"""Object storage contract for isolated intake and parser workers."""
from uuid import UUID
from dataclasses import dataclass
from pathlib import Path
from collections.abc import AsyncIterator
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    uri: str
    byte_size: int
    content_sha256: str
    detected_mime_type: str


class KnowledgeObjectStore(Protocol):
    async def stage_file(
        self, *, receipt_id: str, part_name: str, source_path: Path, expected_sha256: str,
        expected_size: int, detected_mime_type: str,
    ) -> StoredObject: ...

    async def publish_staged(
        self, *, staged: StoredObject, collection_id: UUID, document_id: UUID,
    ) -> StoredObject: ...

    async def delete(self, uri: str) -> None: ...

    async def size(self, uri: str) -> int: ...

    def stream(
        self,
        uri: str,
        *,
        offset: int = 0,
        length: int | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> AsyncIterator[bytes]: ...
