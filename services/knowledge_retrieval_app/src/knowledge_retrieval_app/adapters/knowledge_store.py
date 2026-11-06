"""App 独占的本地不可变对象适配器；只接受摘要句柄，不暴露路径。"""

import asyncio
from collections.abc import AsyncIterator
from hashlib import sha256
import os
from pathlib import Path
import re
import stat

from knowledge_algorithms.contracts.errors import require
from knowledge_retrieval_app.domain.knowledge_ingestion import OWNER, ObjectReceipt


async def _io(operation, *args):
    """取消时等待在途文件操作结束，避免关闭后重用文件描述符。"""
    task = asyncio.create_task(asyncio.to_thread(operation, *args))
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class LocalKnowledgeStore:
    chunk_size = 65536

    def __init__(self, root: Path):
        self.root = root.resolve() / OWNER
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not self.root.is_symlink(), "对象命名空间不能为符号链接")

    def _directory(self, area):
        require(area in {"sources", "staging", "parsed", "representations"}, "对象区域未登记")
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.mkdir(area, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            return os.open(area, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        finally:
            os.close(root_fd)

    def _parts(self, handle):
        require(type(handle) is str and re.fullmatch(r"(sources|parsed|representations)-[0-9a-f]{64}", handle) is not None,
                "对象句柄无效，不接受路径或 URL")
        return handle.split("-", 1)

    async def put(self, blocks: AsyncIterator[bytes], *, area: str, expected_sha256: str,
                  expected_size: int, mime_type: str, max_bytes: int) -> ObjectReceipt:
        require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None
                and type(expected_size) is int and 0 < expected_size <= max_bytes, "对象摘要或大小超限")
        require(area in {"sources", "parsed", "representations"}, "对象区域未登记")
        directory = self._directory(area)
        temporary = "pending-" + os.urandom(16).hex()
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        digest, count = sha256(), 0
        iterator = blocks.__aiter__()
        try:
            async for block in iterator:
                require(type(block) is bytes and 0 < len(block) <= self.chunk_size, "对象流块大小无效")
                count += len(block)
                require(count <= expected_size, "对象流超过声明大小")
                digest.update(block)
                view = memoryview(block)
                while view:
                    written = await _io(os.write, fd, view)
                    require(written > 0, "对象写入未取得进展")
                    view = view[written:]
            require(count == expected_size and digest.hexdigest() == expected_sha256, "对象大小或摘要不匹配")
            await _io(os.fsync, fd)
            os.close(fd)
            fd = None
            try:
                os.link(temporary, expected_sha256, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                os.fsync(directory)
            except FileExistsError:
                existing = os.open(expected_sha256, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    check = sha256()
                    while chunk := os.read(existing, self.chunk_size):
                        check.update(chunk)
                    require(check.hexdigest() == expected_sha256, "不可变对象发生摘要冲突")
                finally:
                    os.close(existing)
            return ObjectReceipt(area + "-" + expected_sha256, expected_sha256, count, mime_type)
        finally:
            if fd is not None:
                os.close(fd)
            os.unlink(temporary, dir_fd=directory)
            os.close(directory)
            close = getattr(iterator, "aclose", None)
            if close:
                await close()

    async def put_bytes(self, value: bytes, *, area="parsed", max_bytes: int):
        async def chunks():
            for offset in range(0, len(value), self.chunk_size):
                yield value[offset:offset + self.chunk_size]
        return await self.put(chunks(), area=area, expected_sha256=sha256(value).hexdigest(),
                              expected_size=len(value), mime_type="application/json", max_bytes=max_bytes)

    async def stream(self, handle: str) -> AsyncIterator[bytes]:
        area, name = self._parts(handle)
        directory = self._directory(area)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        finally:
            os.close(directory)
        try:
            require(stat.S_ISREG(os.fstat(fd).st_mode), "对象不是普通文件")
            digest = sha256()
            while block := await _io(os.read, fd, self.chunk_size):
                digest.update(block)
                yield block
            require(digest.hexdigest() == name, "不可变对象读取摘要不一致")
        finally:
            os.close(fd)

    async def read(self, handle: str, *, max_bytes: int) -> bytes:
        value = bytearray()
        iterator = self.stream(handle)
        try:
            async for block in iterator:
                value.extend(block)
                require(len(value) <= max_bytes, "对象读取超过预算")
        finally:
            await iterator.aclose()
        return bytes(value)
