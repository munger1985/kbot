"""开发与单机部署使用的不可变监控正文存储。"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from uuid import UUID

from aiops_agent.ports.payload_store import StoredMonitorPayload


class LocalMonitorPayloadStore:
    def __init__(self, root: Path):
        self._root = root.expanduser().resolve()

    async def store_verified(
        self, *, source_id: str, body: bytes, content_hash: str
    ) -> StoredMonitorPayload:
        return await asyncio.to_thread(
            self._store_sync, source_id, body, content_hash
        )

    def _store_sync(
        self, source_id: str, body: bytes, content_hash: str
    ) -> StoredMonitorPayload:
        UUID(source_id)
        actual = hashlib.sha256(body).hexdigest()
        if actual != content_hash:
            raise ValueError("监控正文 Hash 与声明不一致")
        target = (
            self._root
            / "verified"
            / source_id
            / content_hash[:2]
            / f"{content_hash}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not target.exists():
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.tmp"
            )
            try:
                with temporary.open("xb") as writer:
                    os.chmod(temporary, 0o600)
                    writer.write(body)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        elif target.stat().st_size != len(body):
            raise RuntimeError("不可变监控正文对象大小冲突")
        return StoredMonitorPayload(
            uri=target.as_uri(),
            content_hash=content_hash,
            byte_size=len(body),
        )

    async def delete(self, uri: str) -> None:
        path = Path(uri.removeprefix("file://"))
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError("拒绝删除监控正文存储根目录之外的对象")
        await asyncio.to_thread(resolved.unlink, missing_ok=True)
