"""复用平台凭据加密；AAD 绑定 Domain、用途与随机凭据身份。"""

from uuid import UUID
from platform_core.managed_credentials.cipher import ManagedCredentialCipher, ManagedCredentialPayload
from platform_core.managed_credentials.entities import ManagedCredentialEntity


class NotificationCredentialVault:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def write(self, *, scope, object_id, values):
        cipher = ManagedCredentialCipher.from_environment()
        payload = cipher.encrypt(values, domain_id=int(scope.domain_id), namespace="notifications",
            credential_kind="feishu", credential_id=object_id)
        async with self._uow_factory() as uow:
            await uow.managed_credentials.add(ManagedCredentialEntity(
                credential_id=object_id, domain_id=int(scope.domain_id), namespace="notifications",
                credential_kind="feishu", external_key=str(object_id), ciphertext=payload.ciphertext,
                nonce=payload.nonce, key_version=payload.key_version, status="ACTIVE",
                created_by="system:notification", updated_by="system:notification"))
            await uow.commit()
        return "managed-credential://notifications/feishu/" + str(object_id)

    async def read(self, *, scope, reference):
        prefix = "managed-credential://notifications/feishu/"
        if not reference.startswith(prefix):
            raise ValueError("NOTIFICATION_CREDENTIAL_REFERENCE_INVALID")
        credential_id = UUID(reference.removeprefix(prefix))
        async with self._uow_factory() as uow:
            row = await uow.managed_credentials.get(domain_id=int(scope.domain_id), credential_id=credential_id)
            if not row or row.namespace != "notifications" or row.credential_kind != "feishu" or row.status != "ACTIVE":
                raise ValueError("NOTIFICATION_CREDENTIAL_UNAVAILABLE")
            payload = ManagedCredentialPayload(row.ciphertext, row.nonce, row.key_version)
        return ManagedCredentialCipher.from_environment().decrypt(payload, domain_id=int(scope.domain_id),
            namespace="notifications", credential_kind="feishu", credential_id=credential_id)

    async def delete(self, *, scope, reference):
        # 错误清理仅使新建凭据失效，不删除仍可能用于审计或任务恢复的历史密文。
        prefix = "managed-credential://notifications/feishu/"
        credential_id = UUID(reference.removeprefix(prefix))
        async with self._uow_factory() as uow:
            row = await uow.managed_credentials.get(domain_id=int(scope.domain_id), credential_id=credential_id, lock=True)
            if row and row.namespace == "notifications" and row.credential_kind == "feishu":
                row.status = "DISABLED"
                await uow.commit()
