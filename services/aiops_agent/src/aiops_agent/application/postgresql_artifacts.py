"""外部 pgBadger Artifact 的有界校验和安全投影。"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import UUID

from platform_core.contracts.aiops.workload import PGBADGER_MAX_UPLOAD_BYTES


_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
_ACTIVE_HTML = re.compile(
    rb"<(?:script|iframe|object|embed|form|input|button|svg|math|base)\b"
    rb"|<meta\b[^>]*http-equiv\s*=\s*[\"']?\s*refresh\b"
    rb"|\bon[a-z0-9_-]+\s*=|\bsrcdoc\s*=|\bstyle\s*=",
    re.IGNORECASE,
)
_EXTERNAL_RESOURCE = re.compile(
    rb"(?:src|href|action|formaction|xlink:href)\s*=\s*[\"']\s*"
    rb"(?:https?:|//|data:|javascript:)",
    re.IGNORECASE,
)


class PgBadgerArtifactRejected(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedPgBadgerArtifact:
    format: str
    content_type: str
    content_hash: str
    byte_size: int
    body: bytes
    trust_level: str = "EXTERNAL_IMPORTED"


class PostgreSQLArtifactStore:
    """只允许在专用根目录保存和读取已校验的PostgreSQL外部报告。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        os.chmod(self._root, 0o700)

    def write(
        self, *, artifact_id: UUID, artifact: ValidatedPgBadgerArtifact
    ) -> str:
        suffix = "json" if artifact.format == "JSON" else "html"
        destination = self._root / f"{artifact_id}.{suffix}"
        temporary = self._root / f".{artifact_id}.{suffix}.tmp"
        if destination.exists():
            if destination.is_symlink():
                raise ValueError("外部报告Artifact目标不能是符号链接")
            existing = destination.read_bytes()
            if hashlib.sha256(existing).hexdigest() != artifact.content_hash:
                raise ValueError("外部报告Artifact标识已存在不同正文")
            return destination.as_uri()
        try:
            with temporary.open("xb") as stream:
                stream.write(artifact.body)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination.as_uri()

    def read(self, payload_uri: str, *, expected_hash: str) -> bytes:
        path = self._path(payload_uri)
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != expected_hash:
            raise ValueError("外部报告Artifact正文Hash不一致")
        return body

    def delete(self, payload_uri: str) -> None:
        self._path(payload_uri).unlink(missing_ok=True)

    def _path(self, payload_uri: str) -> Path:
        parsed = urlparse(payload_uri)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise ValueError("外部报告Artifact地址无效")
        path = Path(unquote(parsed.path)).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("外部报告Artifact地址越界")
        return path


def validate_pgbadger_artifact(
    *, file_name: str, content_type: str, body: bytes
) -> ValidatedPgBadgerArtifact:
    if not body or len(body) > PGBADGER_MAX_UPLOAD_BYTES:
        raise PgBadgerArtifactRejected(
            "PGBADGER_SIZE_INVALID", "pgBadger文件为空或超过20MiB"
        )
    lower_name = file_name.lower()
    decoded = body
    if lower_name.endswith(".gz") or content_type == "application/gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                decoded = stream.read(_MAX_UNCOMPRESSED_BYTES + 1)
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            raise PgBadgerArtifactRejected(
                "PGBADGER_GZIP_INVALID", "pgBadger gzip文件无法解压"
            ) from exc
        lower_name = lower_name.removesuffix(".gz")
    if len(decoded) > _MAX_UNCOMPRESSED_BYTES:
        raise PgBadgerArtifactRejected(
            "PGBADGER_EXPANDED_SIZE_INVALID", "pgBadger解压正文超过100MiB"
        )
    if lower_name.endswith(".json"):
        try:
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PgBadgerArtifactRejected(
                "PGBADGER_JSON_INVALID", "pgBadger JSON正文无效"
            ) from exc
        if not isinstance(payload, dict):
            raise PgBadgerArtifactRejected(
                "PGBADGER_JSON_SHAPE_INVALID", "pgBadger JSON顶层必须是对象"
            )
        normalized = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return _validated("JSON", "application/json", normalized)
    if lower_name.endswith((".html", ".htm")):
        prefix = decoded.lstrip()[:64].lower()
        if not prefix.startswith((b"<!doctype html", b"<html")):
            raise PgBadgerArtifactRejected(
                "PGBADGER_HTML_INVALID", "pgBadger HTML正文缺少HTML根标记"
            )
        if _ACTIVE_HTML.search(decoded) or _EXTERNAL_RESOURCE.search(decoded):
            raise PgBadgerArtifactRejected(
                "PGBADGER_ACTIVE_CONTENT_FORBIDDEN",
                "pgBadger HTML禁止脚本、嵌入对象和外部资源",
            )
        return _validated("HTML", "text/html", decoded)
    raise PgBadgerArtifactRejected(
        "PGBADGER_TYPE_UNSUPPORTED", "仅支持pgBadger JSON、HTML及其gzip文件"
    )


def _validated(
    format_name: str, content_type: str, body: bytes
) -> ValidatedPgBadgerArtifact:
    return ValidatedPgBadgerArtifact(
        format=format_name,
        content_type=content_type,
        content_hash=hashlib.sha256(body).hexdigest(),
        byte_size=len(body),
        body=body,
    )
