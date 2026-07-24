"""Content-addressed local Parser artifact store for development."""

from uuid import UUID
import asyncio
import json
import hashlib
from pathlib import Path
from typing import Any

from knowledge_core.parsing import canonical_json_hash


class LocalParserArtifactStore:
    _ALLOWED = {
        "raw_docling",
        "atom_ir",
        "deepseek_ocr_analysis",
        "structure_ir",
        "evidence_manifest",
        "spreadsheet_artifact",
        "visual_analysis",
    }

    def __init__(self, root: Path):
        self._root = root.resolve()

    async def put_json(
        self, *, job_id: UUID, artifact_name: str, payload: Any,
        expected_sha256: str, schema: str, generator: str,
    ) -> dict[str, str]:
        if artifact_name not in self._ALLOWED:
            raise ValueError(f"unsupported parser artifact: {artifact_name}")
        actual = canonical_json_hash(payload)
        if actual != expected_sha256.lower():
            raise ValueError("parser artifact hash does not match payload")
        target = self._root / "kc-parser-artifacts" / str(job_id) / actual / f"{artifact_name}.json"
        await asyncio.to_thread(self._write_once, target, payload)
        return {
            "uri": str(target), "sha256": actual,
            "schema": schema, "generator": generator,
        }

    async def delete_manifest(self, manifest: dict[str, Any]) -> None:
        for descriptor in manifest.values():
            uri = descriptor.get("uri") if isinstance(descriptor, dict) else None
            if not uri:
                continue
            target = Path(uri).resolve()
            try:
                target.relative_to(self._root)
            except ValueError as exc:
                raise ValueError("parser artifact URI is outside the configured store") from exc
            await asyncio.to_thread(target.unlink, missing_ok=True)

    async def put_bytes(
        self,
        *,
        job_id: UUID,
        asset_key: str,
        payload: bytes,
        expected_sha256: str,
        mime_type: str,
    ) -> dict[str, str]:
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_sha256.lower():
            raise ValueError("视觉资产哈希与正文不一致")
        safe_key = "".join(
            value if value.isalnum() or value in "-_." else "_"
            for value in asset_key
        )[:180]
        target = (
            self._root
            / "kc-parser-assets"
            / str(job_id)
            / actual
            / safe_key
        )
        await asyncio.to_thread(self._write_bytes_once, target, payload)
        return {
            "uri": str(target),
            "sha256": actual,
            "mime_type": mime_type,
        }

    @staticmethod
    def _write_once(target: Path, payload: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        if target.exists():
            if target.read_bytes() != encoded:
                raise ValueError("immutable parser artifact already exists with different content")
            return
        target.write_bytes(encoded)

    @staticmethod
    def _write_bytes_once(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("不可变视觉资产已存在但内容不同")
            return
        target.write_bytes(payload)
