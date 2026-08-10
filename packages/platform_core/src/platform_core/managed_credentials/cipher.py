"""托管凭据的 AES-256-GCM 编解码。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass(frozen=True, slots=True)
class ManagedCredentialPayload:
    ciphertext: bytes
    nonce: bytes
    key_version: str


class ManagedCredentialCipher:
    """使用显式 AAD 绑定凭据所属 Domain 和业务用途。"""

    def __init__(self, *, key: bytes, key_version: str):
        if len(key) != 32:
            raise ValueError("托管凭据主密钥必须是 32 字节")
        if not key_version.strip():
            raise ValueError("托管凭据密钥版本不能为空")
        self._cipher = AESGCM(key)
        self.fingerprint_key = hashlib.sha256(
            b"kbot-managed-credential:fingerprint:" + key
        ).digest()
        self.key_version = key_version.strip()

    @classmethod
    def from_environment(cls) -> "ManagedCredentialCipher":
        raw = os.getenv("KBOT_MANAGED_CREDENTIAL_KEY")
        if not raw:
            raise ValueError("KBOT_MANAGED_CREDENTIAL_KEY 环境变量未设置")
        key = hashlib.sha256(raw.encode("utf-8")).digest()
        version = os.getenv("KBOT_MANAGED_CREDENTIAL_KEY_VERSION", "v1")
        return cls(key=key, key_version=version)

    @staticmethod
    def aad(
        *,
        domain_id: int,
        namespace: str,
        credential_kind: str,
        credential_id: UUID,
    ) -> bytes:
        return (
            f"kbot-managed-credential:v1:{int(domain_id)}:"
            f"{namespace}:{credential_kind}:{credential_id}"
        ).encode("utf-8")

    def encrypt(
        self,
        value: Mapping[str, Any],
        *,
        domain_id: int,
        namespace: str,
        credential_kind: str,
        credential_id: UUID,
    ) -> ManagedCredentialPayload:
        nonce = os.urandom(12)
        plaintext = json.dumps(
            dict(value), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        ciphertext = self._cipher.encrypt(
            nonce,
            plaintext,
            self.aad(
                domain_id=domain_id,
                namespace=namespace,
                credential_kind=credential_kind,
                credential_id=credential_id,
            ),
        )
        return ManagedCredentialPayload(ciphertext, nonce, self.key_version)

    def decrypt(
        self,
        payload: ManagedCredentialPayload,
        *,
        domain_id: int,
        namespace: str,
        credential_kind: str,
        credential_id: UUID,
    ) -> dict[str, Any]:
        if payload.key_version != self.key_version:
            raise ValueError("当前进程不能解密该密钥版本的托管凭据")
        plaintext = self._cipher.decrypt(
            payload.nonce,
            payload.ciphertext,
            self.aad(
                domain_id=domain_id,
                namespace=namespace,
                credential_kind=credential_kind,
                credential_id=credential_id,
            ),
        )
        value = json.loads(plaintext.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("托管凭据载荷必须是 JSON 对象")
        return value


__all__ = ["ManagedCredentialCipher", "ManagedCredentialPayload"]
