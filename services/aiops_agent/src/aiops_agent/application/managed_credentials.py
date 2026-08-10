"""AIOps 统一托管凭据服务。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from platform_core.identity import uuid7
from platform_core.managed_credentials import (
    ManagedCredentialCipher,
    ManagedCredentialEntity,
    ManagedCredentialPayload,
)


class AIOpsManagedCredentialError(ValueError):
    """凭据不存在、作用域不匹配或密文不可解。"""


class AIOpsManagedCredentialService:
    """在平台统一表中管理 AIOps 数据库和监控凭据。"""

    _NAMESPACE = "aiops"
    _PREFIX = "managed-credential://aiops/"

    def __init__(self, *, uow_factory, cipher: ManagedCredentialCipher):
        self._uow_factory = uow_factory
        self._cipher = cipher

    async def put(
        self,
        *,
        uow,
        domain_id: int,
        external_key: UUID,
        credential_kind: str,
        values: dict[str, Any],
        actor_id: str,
    ) -> ManagedCredentialEntity:
        if not values or not all(
            isinstance(key, str)
            and key
            and isinstance(value, str)
            and value
            for key, value in values.items()
        ):
            raise AIOpsManagedCredentialError("AIOps 凭据字段不能为空")
        assert uow.managed_credentials is not None
        row = await uow.managed_credentials.find(
            domain_id=domain_id,
            namespace=self._NAMESPACE,
            credential_kind=credential_kind,
            external_key=str(external_key),
            lock=True,
        )
        credential_id = row.credential_id if row is not None else uuid7()
        encrypted = self._cipher.encrypt(
            values,
            domain_id=domain_id,
            namespace=self._NAMESPACE,
            credential_kind=credential_kind,
            credential_id=credential_id,
        )
        if row is None:
            row = ManagedCredentialEntity(
                credential_id=credential_id,
                domain_id=domain_id,
                namespace=self._NAMESPACE,
                credential_kind=credential_kind,
                external_key=str(external_key),
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

    async def revoke(
        self,
        *,
        uow,
        domain_id: int,
        credential_id: UUID,
        credential_kind: str,
        actor_id: str,
    ) -> None:
        assert uow.managed_credentials is not None
        row = await uow.managed_credentials.get(
            domain_id=domain_id,
            credential_id=credential_id,
            lock=True,
        )
        if row is None:
            return
        if (
            row.namespace != self._NAMESPACE
            or row.credential_kind != credential_kind
        ):
            raise AIOpsManagedCredentialError("AIOps 凭据作用域不匹配")
        row.status = "REVOKED"
        row.updated_by = actor_id

    async def read(
        self,
        *,
        domain_id: int,
        credential_id: UUID,
        credential_kind: str,
        external_key: UUID,
        lock: bool = False,
        uow=None,
    ) -> dict[str, Any]:
        if uow is None:
            async with self._uow_factory() as local_uow:
                value = await self.read(
                    uow=local_uow,
                    domain_id=domain_id,
                    credential_id=credential_id,
                    credential_kind=credential_kind,
                    external_key=external_key,
                    lock=lock,
                )
                await local_uow.commit()
                return value
        assert uow.managed_credentials is not None
        row = await uow.managed_credentials.get(
            domain_id=domain_id,
            credential_id=credential_id,
            lock=lock,
        )
        if (
            row is None
            or row.status != "ACTIVE"
            or row.namespace != self._NAMESPACE
            or row.credential_kind != credential_kind
            or row.external_key != str(external_key)
        ):
            raise AIOpsManagedCredentialError("AIOps 凭据不可用")
        try:
            value = self._cipher.decrypt(
                ManagedCredentialPayload(
                    ciphertext=bytes(row.ciphertext),
                    nonce=bytes(row.nonce),
                    key_version=row.key_version,
                ),
                domain_id=domain_id,
                namespace=self._NAMESPACE,
                credential_kind=credential_kind,
                credential_id=credential_id,
            )
        except Exception as exc:
            raise AIOpsManagedCredentialError("AIOps 凭据不可用") from exc
        return value

    @classmethod
    def reference(
        cls,
        *,
        domain_id: int,
        external_key: UUID,
        credential_kind: str,
        credential_id: UUID,
    ) -> str:
        return (
            f"{cls._PREFIX}{credential_kind}/{int(domain_id)}/"
            f"{external_key}/{credential_id}"
        )

    @classmethod
    def parse_reference(
        cls, reference: str
    ) -> tuple[str, int, UUID, UUID]:
        if not reference.startswith(cls._PREFIX):
            raise AIOpsManagedCredentialError("AIOps 凭据引用无效")
        remainder = reference.removeprefix(cls._PREFIX)
        try:
            kind, raw_domain, raw_external, raw_id = remainder.split("/")
            domain_id = int(raw_domain)
            if domain_id <= 0:
                raise ValueError("domain_id 必须为正整数")
            return kind, domain_id, UUID(raw_external), UUID(raw_id)
        except (ValueError, TypeError) as exc:
            raise AIOpsManagedCredentialError("AIOps 凭据引用无效") from exc

    async def resolve_reference(self, reference: str) -> dict[str, Any]:
        kind, domain_id, external_key, credential_id = self.parse_reference(
            reference
        )
        return await self.read(
            domain_id=domain_id,
            credential_id=credential_id,
            credential_kind=kind,
            external_key=external_key,
        )


__all__ = ["AIOpsManagedCredentialError", "AIOpsManagedCredentialService"]
