"""门户 API Key 的生成、摘要和校验。"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone


API_KEY_PREFIX = "kbot_sk_"
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


class PortalApiKeyError(ValueError):
    """门户 API Key 校验失败。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PortalApiKeyRecord:
    """服务端保存的 API Key 摘要记录。"""

    key_id: str
    client_id: str
    key_digest: str
    enabled: bool = True
    expires_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PortalPrincipal:
    """API Key 校验成功后的调用方身份。"""

    key_id: str
    client_id: str


def digest_portal_api_key(raw_key: str, pepper: str) -> str:
    """使用部署级 Pepper 计算不可逆摘要。"""
    if not raw_key or not pepper:
        raise ValueError("API Key 和 Pepper 不能为空")
    return hmac.new(
        pepper.encode("utf-8"),
        raw_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def generate_portal_api_key(*, key_id: str, pepper: str) -> tuple[str, str]:
    """生成只显示一次的 API Key 及其服务端摘要。"""
    if not _KEY_ID_PATTERN.fullmatch(key_id):
        raise ValueError("key_id 只能包含字母、数字、下划线或连字符")
    raw_key = f"{API_KEY_PREFIX}{key_id}.{secrets.token_urlsafe(32)}"
    return raw_key, digest_portal_api_key(raw_key, pepper)


def extract_bearer_token(authorization: str | None) -> str:
    """从 Authorization Header 提取 Bearer Token。"""
    if not authorization:
        raise PortalApiKeyError("AUTH_REQUIRED", "缺少 Authorization Header")
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise PortalApiKeyError("INVALID_AUTH_SCHEME", "必须使用 Bearer 认证")
    return token.strip()


class PortalApiKeyVerifier:
    """使用内存中的摘要注册表校验门户 API Key。"""

    def __init__(self, *, records: list[PortalApiKeyRecord], pepper: str):
        if not pepper:
            raise ValueError("API Key Pepper 不能为空")
        key_ids = [record.key_id for record in records]
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("Portal API Key 配置存在重复 key_id")
        self._records = {record.key_id: record for record in records}
        self._pepper = pepper
        self._dummy_digest = "0" * 64

    def verify_authorization(
        self,
        authorization: str | None,
        *,
        now: datetime | None = None,
    ) -> PortalPrincipal:
        raw_key = extract_bearer_token(authorization)
        key_id = self._extract_key_id(raw_key)
        record = self._records.get(key_id)
        expected_digest = record.key_digest if record else self._dummy_digest
        actual_digest = digest_portal_api_key(raw_key, self._pepper)
        if record is None or not hmac.compare_digest(actual_digest, expected_digest):
            raise PortalApiKeyError("INVALID_API_KEY", "API Key 无效")
        if not record.enabled:
            raise PortalApiKeyError("API_KEY_DISABLED", "API Key 已停用")
        current_time = now or datetime.now(timezone.utc)
        expires_at = record.expires_at
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if current_time >= expires_at:
                raise PortalApiKeyError("API_KEY_EXPIRED", "API Key 已过期")
        return PortalPrincipal(
            key_id=record.key_id,
            client_id=record.client_id,
        )

    @staticmethod
    def _extract_key_id(raw_key: str) -> str:
        if not raw_key.startswith(API_KEY_PREFIX):
            raise PortalApiKeyError("INVALID_API_KEY", "API Key 无效")
        identifier, separator, secret = raw_key[len(API_KEY_PREFIX):].partition(".")
        if (
            separator != "."
            or not _KEY_ID_PATTERN.fullmatch(identifier)
            or len(secret) < 32
        ):
            raise PortalApiKeyError("INVALID_API_KEY", "API Key 无效")
        return identifier
