"""参数保护库；独立命名空间不能通过普通 AIOps 托管凭据引用兑换。"""

from datetime import datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher, ManagedCredentialEntity, ManagedCredentialPayload


NAMESPACE = "aiops.sql-plan"
KIND = "protected_parameter_material"


class SqlPlanMaterialVault:
    """主密钥仅保留在控制面，Worker/模型/Executor 不持有该保护库的读取入口。"""

    def __init__(self, *, cipher: ManagedCredentialCipher):
        self._cipher = cipher

    @staticmethod
    def owns_run(*, scope, run):
        return (str(run.domain_id) == str(scope.domain_id) and str(run.actor_id) == str(scope.actor_id))

    @staticmethod
    def _scope(scope):
        return {"domain_id": scope.domain_id}

    async def _row(self, *, uow, scope, material_id):
        row = await uow.managed_credentials.find(
            **self._scope(scope), namespace=NAMESPACE, credential_kind=KIND,
            external_key=str(material_id), lock=True,
        )
        if row is None or str(row.created_by) != str(scope.actor_id):
            return None
        return row

    async def seal(self, *, uow, scope, material_id, payload):
        existing = await uow.managed_credentials.find(**self._scope(scope),
            namespace=NAMESPACE, credential_kind=KIND, external_key=str(material_id), lock=True)
        if existing is not None:
            raise state_conflict("保护材料标识不可复用")
        credential_id = uuid7()
        encrypted = self._cipher.encrypt(payload, **self._scope(scope),
            namespace=NAMESPACE, credential_kind=KIND, credential_id=credential_id)
        await uow.managed_credentials.add(ManagedCredentialEntity(
            credential_id=credential_id, **self._scope(scope), namespace=NAMESPACE,
            credential_kind=KIND, external_key=str(material_id), ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce, key_version=encrypted.key_version, status="ACTIVE",
            created_by=scope.actor_id, updated_by=scope.actor_id,
        ))

    def _clear(self, *, row, scope, binding, state):
        # Oracle 的空 RAW 等同 NULL，凭据状态也不是材料状态；以无原值的加密墓碑替换。
        encrypted = self._cipher.encrypt({"binding": binding, "closure": state},
            **self._scope(scope), namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)
        row.status = "REVOKED"
        row.ciphertext = encrypted.ciphertext
        row.nonce = encrypted.nonce
        row.key_version = encrypted.key_version

    async def state(self, *, uow, scope, material_id: UUID, binding, now):
        row = await self._row(uow=uow, scope=scope, material_id=material_id)
        if row is None:
            return "UNAVAILABLE"
        if row.status not in {"ACTIVE", "REVOKED"}:
            return "UNAVAILABLE"
        try:
            protected = self._cipher.decrypt(ManagedCredentialPayload(
                ciphertext=bytes(row.ciphertext), nonce=bytes(row.nonce), key_version=row.key_version,
            ), **self._scope(scope), namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)
            if protected.get("binding") != binding:
                return "UNAVAILABLE"
            if row.status == "REVOKED":
                if set(protected) != {"binding", "closure"} or protected["closure"] not in {"REVOKED", "EXPIRED"}:
                    return "UNAVAILABLE"
                return protected["closure"]
            if set(protected) != {"binding", "document"}:
                return "UNAVAILABLE"
            expiry = datetime.fromisoformat(protected["binding"]["expires_at"])
            if expiry.tzinfo is None:
                return "UNAVAILABLE"
        except (ValueError, TypeError, KeyError):
            return "UNAVAILABLE"
        except Exception:
            # GCM 验证失败也不得将底层异常、密文或参数送到 API/日志。
            return "UNAVAILABLE"
        if now >= expiry:
            self._clear(row=row, scope=scope, binding=binding, state="EXPIRED")
            row.updated_by = scope.actor_id
            return "EXPIRED"
        return "STORED"

    async def open_for_capture(self, *, uow, scope, material_id, binding, now):
        """专用采集服务在当前审批和任务租约核验后调用，不提供公开读入口。"""
        if await self.state(uow=uow, scope=scope, material_id=material_id, binding=binding, now=now) != "STORED":
            raise state_conflict("计划材料不可兑换")
        row = await self._row(uow=uow, scope=scope, material_id=material_id)
        protected = self._cipher.decrypt(ManagedCredentialPayload(
            ciphertext=bytes(row.ciphertext), nonce=bytes(row.nonce), key_version=row.key_version),
            **self._scope(scope), namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)
        return protected["document"]

    async def erase(self, *, uow, scope, material_id, binding):
        row = await self._row(uow=uow, scope=scope, material_id=material_id)
        if row is None:
            raise resource_not_found("SQL 计划材料")
        self._clear(row=row, scope=scope, binding=binding, state="REVOKED")
        row.updated_by = scope.actor_id
