"""Secret 引用校验适配器。"""

from __future__ import annotations

import hashlib
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialError,
    AIOpsManagedCredentialService,
)

from aiops_agent.application.errors import dependency_unavailable, validation_failed
from aiops_agent.ports.secret_store import (
    ResolvedSecret,
    SecretReferenceMetadata,
)


class ConfiguredSecretStore:
    """仅解析数据库中的 AIOps 托管凭据引用。"""

    def __init__(self, *, managed_credentials: AIOpsManagedCredentialService):
        self._managed_credentials = managed_credentials

    async def validate_ref(self, reference: str) -> SecretReferenceMetadata:
        try:
            await self._managed_credentials.resolve_reference(reference)
        except AIOpsManagedCredentialError as exc:
            raise validation_failed("AIOps 托管凭据引用无效") from exc
        return SecretReferenceMetadata(
            provider="managed-credential",
            fingerprint=hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest()[:16],
        )

    async def resolve(self, reference: str) -> ResolvedSecret:
        """从平台统一凭据表解密字符串键值。"""
        try:
            decoded = await self._managed_credentials.resolve_reference(
                reference
            )
        except AIOpsManagedCredentialError as exc:
            raise dependency_unavailable("AIOps 托管凭据不可用") from exc
        if not isinstance(decoded, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in decoded.items()
        ):
            raise validation_failed("Secret 必须是字符串或字符串 JSON 对象")
        if not decoded:
            raise validation_failed("Secret 不能为空")
        return ResolvedSecret(
            values=dict(decoded),
            fingerprint=hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest()[:16],
        )
