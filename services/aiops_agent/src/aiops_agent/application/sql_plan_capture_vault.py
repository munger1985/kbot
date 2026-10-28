"""审批冻结记录的独立密文；普通 Artifact 哈希不能为自身证明审批。"""

from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialEntity, ManagedCredentialPayload

from aiops_agent.application.errors import state_conflict
from aiops_agent.application.sql_plan_material_vault import NAMESPACE


KIND = "plan_capture_authorization"


class SqlPlanCaptureVault:
    """只保护安全审批元数据，不读取代表参数；锁序为 Run 后凭据行。"""

    def __init__(self, *, material_vault):
        self._materials = material_vault

    async def get(self, *, uow, scope, request_id, lock=True):
        row = await uow.managed_credentials.find(**self._materials._scope(scope),
            namespace=NAMESPACE, credential_kind=KIND, external_key=str(request_id), lock=lock)
        if row is None or row.status not in {"ACTIVE", "REVOKED"}:
            return None, None
        try:
            payload = self._materials._cipher.decrypt(ManagedCredentialPayload(
                ciphertext=bytes(row.ciphertext), nonce=bytes(row.nonce), key_version=row.key_version),
                **self._materials._scope(scope), namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)
            request = payload["request"]
            if (str(request["request_id"]) != str(request_id) or str(request["requested_by"]) != str(row.created_by)
                or payload["state"] not in {"PENDING", "APPROVED", "REJECTED", "REVOKED"}
                or (row.status == "REVOKED") != (payload["state"] in {"REJECTED", "REVOKED"})):
                return None, None
            return row, payload
        except Exception:
            # 不把密文、身份或底层异常放进响应及日志。
            return None, None

    async def create(self, *, uow, scope, request):
        if await uow.managed_credentials.find(**self._materials._scope(scope), namespace=NAMESPACE,
            credential_kind=KIND, external_key=request["request_id"], lock=True) is not None:
            raise state_conflict("计划采集请求标识不可复用")
        credential_id = uuid7()
        payload = {"request": request, "state": "PENDING", "decision": None}
        encrypted = self._materials._cipher.encrypt(payload, **self._materials._scope(scope),
            namespace=NAMESPACE, credential_kind=KIND, credential_id=credential_id)
        await uow.managed_credentials.add(ManagedCredentialEntity(credential_id=credential_id,
            **self._materials._scope(scope), namespace=NAMESPACE, credential_kind=KIND,
            external_key=request["request_id"], ciphertext=encrypted.ciphertext,
            nonce=encrypted.nonce, key_version=encrypted.key_version, status="ACTIVE",
            created_by=scope.actor_id, updated_by=scope.actor_id))
        return payload

    def update(self, *, row, scope, payload):
        encrypted = self._materials._cipher.encrypt(payload, **self._materials._scope(scope),
            namespace=NAMESPACE, credential_kind=KIND, credential_id=row.credential_id)
        row.ciphertext, row.nonce, row.key_version = encrypted.ciphertext, encrypted.nonce, encrypted.key_version
        row.status = "REVOKED" if payload["state"] in {"REJECTED", "REVOKED"} else "ACTIVE"
        row.updated_by = scope.actor_id
