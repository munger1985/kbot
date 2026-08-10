"""Data Query 数据库凭据的统一托管服务。"""

from __future__ import annotations

from uuid import UUID

from platform_core.identity import uuid7
from platform_core.managed_credentials import (
    ManagedCredentialCipher,
    ManagedCredentialEntity,
    ManagedCredentialPayload,
)


class DataQueryCredentialError(ValueError):
    """数据库凭据不存在、边界不匹配或密文不可解。"""


class DatabaseCredentialService:
    """通过平台统一凭据表写入和读取 Data Query 数据库账号。"""

    _NAMESPACE = "data_query"
    _KIND = "database"

    def __init__(self, *, uow_factory, cipher: ManagedCredentialCipher):
        self._uow_factory = uow_factory
        self._cipher = cipher

    async def create(
        self, *, uow, domain_id: int, data_source_id: UUID,
        username: str, password: str, actor_id: str,
    ) -> ManagedCredentialEntity:
        assert uow.managed_credentials
        external_key = str(data_source_id)
        row = await uow.managed_credentials.find(
            domain_id=domain_id,
            namespace=self._NAMESPACE,
            credential_kind=self._KIND,
            external_key=external_key,
            lock=True,
        )
        credential_id = row.credential_id if row is not None else uuid7()
        encrypted = self._cipher.encrypt(
            {"username": username, "password": password},
            domain_id=domain_id,
            namespace=self._NAMESPACE,
            credential_kind=self._KIND,
            credential_id=credential_id,
        )
        if row is None:
            row = ManagedCredentialEntity(
                credential_id=credential_id,
                domain_id=domain_id,
                namespace=self._NAMESPACE,
                credential_kind=self._KIND,
                external_key=external_key,
                ciphertext=encrypted.ciphertext,
                nonce=encrypted.nonce,
                key_version=encrypted.key_version,
                status="ACTIVE",
                created_by=actor_id,
                updated_by=actor_id,
            )
            await uow.managed_credentials.add(row)
        else:
            row.ciphertext = encrypted.ciphertext
            row.nonce = encrypted.nonce
            row.key_version = encrypted.key_version
            row.status = "ACTIVE"
            row.updated_by = actor_id
        return row

    async def read_database_credentials(
        self, *, credential_id: UUID, domain_id: int, data_source_id: UUID,
    ) -> tuple[str, str]:
        async with self._uow_factory() as uow:
            assert uow.managed_credentials
            row = await uow.managed_credentials.get(
                domain_id=domain_id, credential_id=credential_id
            )
            await uow.commit()
        if (
            row is None
            or row.status != "ACTIVE"
            or row.namespace != self._NAMESPACE
            or row.credential_kind != self._KIND
            or row.external_key != str(data_source_id)
        ):
            raise DataQueryCredentialError("Data Query 凭据不可用")
        try:
            value = self._cipher.decrypt(
                ManagedCredentialPayload(
                    ciphertext=row.ciphertext,
                    nonce=row.nonce,
                    key_version=row.key_version,
                ),
                domain_id=domain_id,
                namespace=self._NAMESPACE,
                credential_kind=self._KIND,
                credential_id=credential_id,
            )
            username = value.get("username")
            password = value.get("password")
            if not isinstance(username, str) or not isinstance(password, str):
                raise ValueError("凭据字段缺失")
            return username, password
        except Exception as exc:
            raise DataQueryCredentialError("Data Query 凭据不可用") from exc


__all__ = ["DataQueryCredentialError", "DatabaseCredentialService"]
