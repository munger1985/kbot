"""Storage boundary for immutable Parser output artifacts."""

from uuid import UUID
from typing import Any, Protocol


class ParserArtifactStore(Protocol):
    async def put_json(
        self, *, job_id: UUID, artifact_name: str, payload: Any,
        expected_sha256: str, schema: str, generator: str,
    ) -> dict[str, str]: ...

    async def put_bytes(
        self,
        *,
        job_id: UUID,
        asset_key: str,
        payload: bytes,
        expected_sha256: str,
        mime_type: str,
    ) -> dict[str, str]: ...

    async def delete_manifest(self, manifest: dict[str, Any]) -> None: ...
