"""Content-addressed cache for successful OCR/VLM visual results."""

import asyncio
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from knowledge_core.parsing import canonical_json_hash


class LocalVisualResultCache:
    def __init__(self, root: Path):
        self._root = root.resolve()

    async def get(
        self, *, image, kind: str, model: str, prompt: str,
        options: dict[str, Any],
    ) -> str | None:
        key = self._key(
            image=image, kind=kind, model=model, prompt=prompt,
            options=options,
        )
        return await asyncio.to_thread(self._read, key)

    async def put(
        self, *, image, kind: str, model: str, prompt: str,
        options: dict[str, Any], result: str,
    ) -> None:
        text = result.strip()
        if not text:
            return
        key = self._key(
            image=image, kind=kind, model=model, prompt=prompt,
            options=options,
        )
        await asyncio.to_thread(self._write, key, text)

    @classmethod
    def image_sha256(cls, image) -> str:
        return sha256(cls._image_bytes(image)).hexdigest()

    def _key(
        self, *, image, kind: str, model: str, prompt: str,
        options: dict[str, Any],
    ) -> str:
        return canonical_json_hash({
            "image_sha256": self.image_sha256(image),
            "kind": kind,
            "model": model,
            "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "options": options,
        })

    @staticmethod
    def _image_bytes(image) -> bytes:
        stream = BytesIO()
        image.convert("RGB").save(stream, format="PNG")
        return stream.getvalue()

    def _read(self, key: str) -> str | None:
        target = self._root / key[:2] / f"{key}.json"
        if not target.is_file():
            return None
        payload = json.loads(target.read_text(encoding="utf-8"))
        result = payload.get("result")
        return result if isinstance(result, str) and result.strip() else None

    def _write(self, key: str, result: str) -> None:
        target = self._root / key[:2] / f"{key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            {"schema": "kc-visual-cache/v1", "result": result},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if target.exists():
            return
        temporary = target.with_suffix(f".{uuid4().hex}.tmp")
        try:
            temporary.write_text(encoded, encoding="utf-8")
            if not target.exists():
                temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
