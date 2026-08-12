"""KM Asset App 托管凭据服务。"""

from typing import Any
from uuid import UUID

from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher, ManagedCredentialEntity, ManagedCredentialPayload


class KmCredentialService:
    NAMESPACE = "km_asset"

    def __init__(self, *, cipher: ManagedCredentialCipher):
        self._cipher = cipher

    async def put(self, *, uow, domain_id: int, source_id: UUID, credential_kind: str, values: dict[str, str], actor_id: str):
        if not values or any(not str(key).strip() or not str(value).strip() for key, value in values.items()):
            raise ValueError("KM Asset 凭据字段不能为空")
        row = await uow.managed_credentials.find(domain_id=domain_id, namespace=self.NAMESPACE, credential_kind=credential_kind, external_key=str(source_id), lock=True)
        credential_id = row.credential_id if row is not None else uuid7()
        encrypted = self._cipher.encrypt(values, domain_id=domain_id, namespace=self.NAMESPACE, credential_kind=credential_kind, credential_id=credential_id)
        if row is None:
            row = ManagedCredentialEntity(credential_id=credential_id, domain_id=domain_id, namespace=self.NAMESPACE, credential_kind=credential_kind, external_key=str(source_id), ciphertext=encrypted.ciphertext, nonce=encrypted.nonce, key_version=encrypted.key_version, status="ACTIVE", created_by=actor_id, updated_by=actor_id)
            await uow.managed_credentials.add(row)
        else:
            row.ciphertext = encrypted.ciphertext
            row.nonce = encrypted.nonce
            row.key_version = encrypted.key_version
            row.status = "ACTIVE"
            row.updated_by = actor_id
        return row

    async def read(self, *, uow, domain_id: int, credential_id: UUID, credential_kind: str, source_id: UUID) -> dict[str, Any]:
        row = await uow.managed_credentials.get(domain_id=domain_id, credential_id=credential_id)
        if row is None or row.status != "ACTIVE" or row.namespace != self.NAMESPACE or row.credential_kind != credential_kind or row.external_key != str(source_id):
            raise ValueError("KM Asset 凭据不存在或作用域不匹配")
        return self._cipher.decrypt(ManagedCredentialPayload(ciphertext=row.ciphertext, nonce=row.nonce, key_version=row.key_version), domain_id=domain_id, namespace=self.NAMESPACE, credential_kind=credential_kind, credential_id=credential_id)
