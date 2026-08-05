"""Data Query 数据库凭据的加密与最小暴露服务。"""

from __future__ import annotations

import base64
import binascii
import os
import secrets
from dataclasses import dataclass
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from data_query.entities import CredentialEntity
from platform_core.identity import uuid7


class DataQueryCredentialError(ValueError):
    """凭据配置、密文或认证标签无效。"""


@dataclass(frozen=True)
class EncryptedCredential:
    username_ciphertext: bytes
    username_nonce: bytes
    password_ciphertext: bytes
    password_nonce: bytes
    key_version: str


class CredentialCipher:
    """使用 Data Query 独立 AES-256-GCM 密钥保护数据库账号。"""

    def __init__(self, *, key: bytes, key_version: str):
        if len(key) != 32 or not key_version.strip():
            raise DataQueryCredentialError("Data Query 凭据加密配置无效")
        self._cipher = AESGCM(key)
        self.key_version = key_version

    @classmethod
    def from_environment(cls) -> "CredentialCipher":
        raw = os.getenv("KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY", "")
        try:
            key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
        except (ValueError, binascii.Error) as exc:
            raise DataQueryCredentialError(
                "KBOT_DATA_QUERY_CREDENTIAL_ENCRYPTION_KEY 必须是 Base64URL 编码的 32 字节密钥"
            ) from exc
        return cls(
            key=key,
            key_version=os.getenv(
                "KBOT_DATA_QUERY_CREDENTIAL_KEY_VERSION", "v1"
            ),
        )

    @staticmethod
    def _aad(
        *, domain_id: int, data_source_id: UUID,
        credential_version: int, field: str,
    ) -> bytes:
        return (
            f"kbot:data-query:{domain_id}:{data_source_id}:"
            f"{credential_version}:{field}"
        ).encode("utf-8")

    def encrypt(
        self, *, domain_id: int, data_source_id: UUID,
        credential_version: int, username: str, password: str,
    ) -> EncryptedCredential:
        username_nonce = secrets.token_bytes(12)
        password_nonce = secrets.token_bytes(12)
        return EncryptedCredential(
            username_ciphertext=self._cipher.encrypt(
                username_nonce,
                username.encode("utf-8"),
                self._aad(
                    domain_id=domain_id,
                    data_source_id=data_source_id,
                    credential_version=credential_version,
                    field="username",
                ),
            ),
            username_nonce=username_nonce,
            password_ciphertext=self._cipher.encrypt(
                password_nonce,
                password.encode("utf-8"),
                self._aad(
                    domain_id=domain_id,
                    data_source_id=data_source_id,
                    credential_version=credential_version,
                    field="password",
                ),
            ),
            password_nonce=password_nonce,
            key_version=self.key_version,
        )

    def decrypt(
        self, *, domain_id: int, data_source_id: UUID,
        credential_version: int, row: CredentialEntity,
    ) -> tuple[str, str]:
        if row.key_version != self.key_version:
            raise DataQueryCredentialError("Data Query 凭据不可用")
        try:
            return (
                self._cipher.decrypt(
                    row.username_nonce,
                    row.username_ciphertext,
                    self._aad(
                        domain_id=domain_id,
                        data_source_id=data_source_id,
                        credential_version=credential_version,
                        field="username",
                    ),
                ).decode("utf-8"),
                self._cipher.decrypt(
                    row.password_nonce,
                    row.password_ciphertext,
                    self._aad(
                        domain_id=domain_id,
                        data_source_id=data_source_id,
                        credential_version=credential_version,
                        field="password",
                    ),
                ).decode("utf-8"),
            )
        except Exception as exc:
            raise DataQueryCredentialError("Data Query 凭据不可用") from exc


class DatabaseCredentialService:
    """在调用方 UoW 内写密文，并在执行时按 Domain 重读。"""

    def __init__(self, *, uow_factory, cipher: CredentialCipher):
        self._uow_factory = uow_factory
        self._cipher = cipher

    async def create(
        self, *, uow, domain_id: int, data_source_id: UUID,
        credential_version: int, username: str, password: str, actor_id: str,
    ) -> CredentialEntity:
        credential_id = uuid7()
        encrypted = self._cipher.encrypt(
            domain_id=domain_id,
            data_source_id=data_source_id,
            credential_version=credential_version,
            username=username,
            password=password,
        )
        row = CredentialEntity(
            credential_id=credential_id,
            domain_id=domain_id,
            data_source_id=data_source_id,
            credential_version=credential_version,
            username_ciphertext=encrypted.username_ciphertext,
            username_nonce=encrypted.username_nonce,
            password_ciphertext=encrypted.password_ciphertext,
            password_nonce=encrypted.password_nonce,
            key_version=encrypted.key_version,
            status="ACTIVE",
            created_by=actor_id,
            updated_by=actor_id,
        )
        await uow.credentials.add(row)
        return row

    async def read_database_credentials(
        self, *, credential_id: UUID, domain_id: int, data_source_id: UUID,
    ) -> tuple[str, str]:
        async with self._uow_factory() as uow:
            row = await uow.credentials.get_scoped(
                credential_id=credential_id,
                domain_id=domain_id,
                data_source_id=data_source_id,
                active_only=True,
            )
            await uow.commit()
        if row is None:
            raise DataQueryCredentialError("Data Query 凭据不可用")
        return self._cipher.decrypt(
            domain_id=domain_id,
            data_source_id=data_source_id,
            credential_version=int(row.credential_version),
            row=row,
        )
