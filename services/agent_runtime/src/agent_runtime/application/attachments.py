"""Conversation 查询附件的不可变本地存储适配器。"""

import asyncio
import base64
import hashlib
import os
import secrets
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
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError("单张查询图片超过 8 MiB")
            if total > 8 * 1024 * 1024:
                raise ValueError("查询图片总大小超过 8 MiB")
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
        if target.parent.resolve() != target.parent.absolute():
            raise ValueError("查询图片目录不能包含符号链接")
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        temporary = ".pending-" + secrets.token_hex(16)
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target.name, src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False)
            except FileExistsError:
                descriptor = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                with os.fdopen(descriptor, "rb") as stream:
                    if stream.read(len(payload) + 1) != payload:
                        raise ValueError("不可变查询图片已存在但内容不同")
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            finally:
                os.close(directory)

    async def delete_conversation(self, conversation_id: UUID) -> None:
        target = (
            self._root / "conversation-query-images" / str(conversation_id)
        ).resolve()
        target.relative_to(self._root)
        await asyncio.to_thread(shutil.rmtree, target, True)
