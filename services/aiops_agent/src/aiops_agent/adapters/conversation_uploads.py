"""AIOps 对话输入文件的本地受控暂存。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import unquote, urlparse
from uuid import UUID

from platform_core.identity import uuid7


@dataclass(frozen=True, slots=True)
class StoredConversationUpload:
    upload_id: str
    domain_id: int
    actor_id: str
    file_name: str
    media_type: str
    byte_size: int
    content_hash: str
    payload_uri: str
    expires_at: datetime
    preserved: bool = False


class LocalConversationUploadStore:
    """以随机 ID 暂存有界输入，并在读取时校验租户与用户归属。"""

    _ALLOWED_MEDIA_TYPES = frozenset(
        {
            "text/plain",
            "text/csv",
            "application/json",
            "application/sql",
            "image/png",
            "image/jpeg",
            "image/webp",
        }
    )

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int,
        ttl_seconds: int = 86_400,
    ) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)
        self._artifact_root = self._root / "artifacts"
        self._artifact_root.mkdir(exist_ok=True)
        os.chmod(self._artifact_root, 0o700)
        self._max_bytes = max_bytes
        self._ttl = timedelta(seconds=ttl_seconds)

    async def store(
        self,
        *,
        domain_id: int,
        actor_id: str,
        file_name: str,
        media_type: str,
        chunks: AsyncIterator[bytes],
    ) -> StoredConversationUpload:
        normalized_media_type = media_type.split(";", 1)[0].strip().lower()
        if normalized_media_type not in self._ALLOWED_MEDIA_TYPES:
            raise ValueError("仅支持文本、JSON、CSV、SQL 和 PNG/JPEG/WebP 图片")
        safe_name = Path(file_name).name.strip()[:256]
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("上传文件名无效")
        upload_id = str(uuid7())
        payload_path = self._root / f"{upload_id}.bin"
        metadata_path = self._root / f"{upload_id}.json"
        temporary_path = self._root / f".{upload_id}.tmp"
        digest = hashlib.sha256()
        byte_size = 0
        try:
            with temporary_path.open("xb") as stream:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    byte_size += len(chunk)
                    if byte_size > self._max_bytes:
                        raise ValueError("上传文件超过允许的大小")
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if byte_size == 0:
                raise ValueError("上传文件不能为空")
            temporary_path.replace(payload_path)
            expires_at = datetime.now(UTC) + self._ttl
            metadata = {
                "upload_id": upload_id,
                "domain_id": domain_id,
                "actor_id": actor_id,
                "file_name": safe_name,
                "media_type": normalized_media_type,
                "byte_size": byte_size,
                "content_hash": digest.hexdigest(),
                "payload_uri": payload_path.as_uri(),
                "expires_at": expires_at.isoformat(),
                "preserved": False,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            os.chmod(payload_path, 0o600)
            os.chmod(metadata_path, 0o600)
            return StoredConversationUpload(
                **{**metadata, "expires_at": expires_at}
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            if not metadata_path.exists():
                payload_path.unlink(missing_ok=True)
            raise

    def get(
        self, *, upload_id: str, domain_id: int, actor_id: str
    ) -> StoredConversationUpload:
        try:
            normalized_id = str(UUID(str(upload_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("上传文件引用无效") from exc
        metadata_path = self._root / f"{normalized_id}.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored = StoredConversationUpload(
                **{
                    **metadata,
                    "expires_at": datetime.fromisoformat(metadata["expires_at"]),
                }
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError("上传文件不存在或元数据损坏") from exc
        if stored.domain_id != domain_id or stored.actor_id != actor_id:
            raise PermissionError("上传文件不属于当前用户")
        if not stored.preserved and stored.expires_at <= datetime.now(UTC):
            raise ValueError("上传文件引用已过期，请重新上传")
        payload_path = self._payload_path(stored.payload_uri)
        if not payload_path.is_file():
            raise ValueError("上传文件正文不存在")
        return stored

    def read(self, stored: StoredConversationUpload) -> bytes:
        payload_path = self._payload_path(stored.payload_uri)
        content = payload_path.read_bytes()
        if len(content) != stored.byte_size:
            raise ValueError("上传文件大小与登记信息不一致")
        if hashlib.sha256(content).hexdigest() != stored.content_hash:
            raise ValueError("上传文件完整性校验失败")
        return content

    def preserve(
        self, stored: StoredConversationUpload
    ) -> StoredConversationUpload:
        """把已被 Turn 接收的正文移入长期 Artifact 区并更新引用。"""
        source = self._payload_path(stored.payload_uri)
        destination = self._artifact_root / f"{stored.upload_id}.bin"
        if source != destination:
            if source.exists():
                source.replace(destination)
            elif not destination.exists():
                raise ValueError("上传文件正文不存在")
        preserved = replace(
            stored, payload_uri=destination.as_uri(), preserved=True
        )
        metadata = asdict(preserved)
        metadata["expires_at"] = preserved.expires_at.isoformat()
        metadata_path = self._root / f"{stored.upload_id}.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.chmod(metadata_path, 0o600)
        return preserved

    def _payload_path(self, payload_uri: str) -> Path:
        parsed = urlparse(payload_uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("上传文件存储地址无效")
        payload_path = Path(unquote(parsed.path)).resolve()
        if not payload_path.is_relative_to(self._root):
            raise ValueError("上传文件存储地址越界")
        return payload_path


__all__ = ["LocalConversationUploadStore", "StoredConversationUpload"]
