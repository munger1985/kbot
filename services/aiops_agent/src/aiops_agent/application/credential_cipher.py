"""AIOps 数据库凭据的最小暴露加密边界。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from dataclasses import dataclass
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CredentialCipherError(ValueError):
    """凭据密文、密钥或认证标签无效时的统一错误。"""


@dataclass(frozen=True)
class EncryptedCredential:
    username_ciphertext: bytes
    username_nonce: bytes
    password_ciphertext: bytes
    password_nonce: bytes
    key_version: str


class CredentialCipher:
    """使用与身份认证隔离的 AES-256-GCM 密钥保护数据库账号。"""

    def __init__(self, *, key: bytes, key_version: str):
        if len(key) != 32 or not key_version:
            raise CredentialCipherError("凭据加密配置无效")
        self._aesgcm = AESGCM(key)
        self._fingerprint_key = key
        self.key_version = key_version

    @classmethod
    def from_environment(cls) -> "CredentialCipher":
        raw = os.getenv("KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY", "")
        try:
            key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        except (ValueError, binascii.Error) as exc:
            raise CredentialCipherError(
                "KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY 必须为 Base64URL 编码的 32 字节密钥"
            ) from exc
        try:
            return cls(
                key=key,
                key_version=os.getenv("KBOT_AIOPS_CREDENTIAL_KEY_VERSION", "v1"),
            )
        except CredentialCipherError as exc:
            raise CredentialCipherError(
                "KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY 必须为 Base64URL 编码的 32 字节密钥"
            ) from exc

    @staticmethod
    def _aad(domain_id: int, credential_id: UUID, credential_kind: str) -> bytes:
        return f"{domain_id}:{credential_id}:{credential_kind}".encode("utf-8")

    def encrypt(
        self, *, domain_id: int, credential_id: UUID, credential_kind: str,
        username: str, password: str,
    ) -> EncryptedCredential:
        aad = self._aad(domain_id, credential_id, credential_kind)
        username_nonce, password_nonce = secrets.token_bytes(12), secrets.token_bytes(12)
        if len(username_nonce) != 12 or len(password_nonce) != 12:
            raise CredentialCipherError("凭据随机 nonce 长度无效")
        return EncryptedCredential(
            username_ciphertext=self._aesgcm.encrypt(username_nonce, username.encode(), aad),
            username_nonce=username_nonce,
            password_ciphertext=self._aesgcm.encrypt(password_nonce, password.encode(), aad),
            password_nonce=password_nonce,
            key_version=self.key_version,
        )

    def decrypt(self, *, domain_id: int, credential_id: UUID, credential_kind: str,
                username_ciphertext: bytes, username_nonce: bytes,
                password_ciphertext: bytes, password_nonce: bytes,
                key_version: str) -> tuple[str, str]:
        if key_version != self.key_version:
            raise CredentialCipherError("凭据不可用")
        try:
            aad = self._aad(domain_id, credential_id, credential_kind)
            return (
                self._aesgcm.decrypt(username_nonce, username_ciphertext, aad).decode(),
                self._aesgcm.decrypt(password_nonce, password_ciphertext, aad).decode(),
            )
        except Exception as exc:
            raise CredentialCipherError("凭据不可用") from exc
