"""Storage boundary for immutable Parser output artifacts."""

from typing import Any, Protocol


class ParserArtifactStore(Protocol):
    async def put_json(
        self, *, job_id: int, artifact_name: str, payload: Any,
        expected_sha256: str, schema: str, generator: str,
    ) -> dict[str, str]: ...

    async def delete_manifest(self, manifest: dict[str, Any]) -> None: ...
