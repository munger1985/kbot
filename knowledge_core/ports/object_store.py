"""Object storage contract for isolated intake and parser workers."""
from dataclasses import dataclass
from pathlib import Path
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
        self, *, staged: StoredObject, collection_id: int, document_id: int,
    ) -> StoredObject: ...

    async def delete(self, uri: str) -> None: ...
