"""多媒体创作工作台本地对象存储：数据库只保存元数据，字节写独立根目录。"""

from __future__ import annotations

import asyncio
from pathlib import Path


class MediaStudioLocalObjectStore:
    """语义对齐 KC 本地存储，但根目录与 KC 完全独立。"""

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()

    async def put(self, object_key: str, content: bytes) -> str:
        path = self._scoped_target(object_key)
        await asyncio.to_thread(self._write_sync, path, content)
        return object_key

    async def get(self, object_key: str) -> bytes:
        path = self._scoped_file(object_key)
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, object_key: str) -> None:
        path = self._scoped_target(object_key)
        await asyncio.to_thread(self._delete_sync, path)

    def _scoped_target(self, object_key: str) -> Path:
        relative = self._relative_key(object_key)
        path = (self._root / relative).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("对象键超出多媒体创作工作台本地存储根目录")
        return path

    def _scoped_file(self, object_key: str) -> Path:
        path = self._scoped_target(object_key)
        if not path.is_file():
            raise FileNotFoundError("对象不存在于多媒体创作工作台本地存储")
        return path

    @staticmethod
    def _relative_key(object_key: str) -> Path:
        key = str(object_key or "").strip().replace("\\", "/")
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("对象键必须是相对路径")
        return Path(key)

    def _write_sync(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    @staticmethod
    def _delete_sync(path: Path) -> None:
        if path.is_file():
            path.unlink()
