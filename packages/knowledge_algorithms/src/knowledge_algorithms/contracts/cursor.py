"""签名续取绑定；游标不是授权，App 使用前仍复核当前权限。"""

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json

from .errors import AlgorithmError, ErrorCode, require
from .identity import canonical_json
from .wire import decode, encode


@dataclass(frozen=True, slots=True)
class CursorBinding:
    owner_binding_ref: str
    subject_ref: str
    scope_hash: str
    snapshot_ref: str
    query_hash: str
    profile_hash: str
    role: str
    mode: str
    order_json: str
    position_json: str
    expires_at: int
    schema_version: str = "knowledge.cursor.v1"

    def __post_init__(self) -> None:
        require(self.schema_version == "knowledge.cursor.v1", "游标版本无效")
        require(type(self.expires_at) is int and self.expires_at > 0, "游标期限无效")
        for text in (self.order_json, self.position_json):
            require(canonical_json(json.loads(text)) == text, "游标位置必须为规范 JSON")


class CursorCodec:
    def __init__(self, secret: bytes, *, max_bytes: int):
        require(type(secret) is bytes and len(secret) >= 32 and max_bytes > 0, "游标签名配置无效")
        self._secret, self.max_bytes = secret, max_bytes

    def encode(self, binding: CursorBinding) -> str:
        payload = canonical_json(encode(binding)).encode("utf-8")
        require(len(payload) <= self.max_bytes, "游标超过大小上限")
        signature = hmac.digest(self._secret, payload, hashlib.sha256)
        return base64.urlsafe_b64encode(signature + payload).decode("ascii")

    def decode(self, token: str, *, expected: CursorBinding, now: int) -> CursorBinding:
        try:
            require(len(token) <= (self.max_bytes + 32) * 2, "游标超过大小上限")
            raw = base64.b64decode(token, altchars=b"-_", validate=True)
            signature, payload = raw[:32], raw[32:]
            require(len(payload) <= self.max_bytes and hmac.compare_digest(signature,
                    hmac.digest(self._secret, payload, hashlib.sha256)), "游标签名无效")
            result = decode(CursorBinding, json.loads(payload))
            if result.expires_at <= now:
                raise AlgorithmError(ErrorCode.CURSOR_EXPIRED, "游标已过期", stage="cursor")
            for name in ("owner_binding_ref", "subject_ref", "scope_hash", "snapshot_ref", "query_hash",
                         "profile_hash", "role", "mode", "order_json", "expires_at"):
                require(getattr(result, name) == getattr(expected, name), "游标上下文不匹配")
            return result
        except AlgorithmError as exc:
            if exc.code == ErrorCode.CURSOR_EXPIRED:
                raise
            raise AlgorithmError(ErrorCode.CURSOR_INVALID, "游标无效", stage="cursor") from exc
        except (ValueError, TypeError, UnicodeError) as exc:
            raise AlgorithmError(ErrorCode.CURSOR_INVALID, "游标无效", stage="cursor") from exc
