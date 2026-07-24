"""Conversation 查询附件的不可变本地存储适配器。"""

import asyncio
import base64
import hashlib
import shutil
from pathlib import Path
from typing import Any
from uuid import UUID


class ConversationAttachmentStore:
    def __init__(self, root: Path):
        self._root = root.resolve()

    async def put_images(
        self, *, conversation_id: UUID, images: tuple[Any, ...]
    ) -> tuple[dict[str, Any], ...]:
        output = []
        total = 0
        for image in images:
            try:
                payload = base64.b64decode(
                    image.content_base64.split(",", 1)[-1], validate=True
                )
            except ValueError as exc:
                raise ValueError("查询图片不是合法 Base64") from exc
            total += len(payload)
            if len(payload) > 16 * 1024 * 1024:
                raise ValueError("单张查询图片超过 16 MiB")
            if total > 32 * 1024 * 1024:
                raise ValueError("查询图片总大小超过 32 MiB")
            digest = hashlib.sha256(payload).hexdigest()
            suffix = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/webp": ".webp",
            }[image.mime_type]
            target = (
                self._root
                / "conversation-query-images"
                / str(conversation_id)
                / f"{digest}{suffix}"
            )
            await asyncio.to_thread(self._write_once, target, payload)
            output.append(
                {
                    "file_name": image.file_name,
                    "mime_type": image.mime_type,
                    "content_sha256": digest,
                    "byte_size": len(payload),
                    "storage_uri": str(target),
                }
            )
        return tuple(output)

    @staticmethod
    def _write_once(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("不可变查询图片已存在但内容不同")
            return
        target.write_bytes(payload)

    async def delete_conversation(self, conversation_id: UUID) -> None:
        target = (
            self._root / "conversation-query-images" / str(conversation_id)
        ).resolve()
        target.relative_to(self._root)
        await asyncio.to_thread(shutil.rmtree, target, True)
