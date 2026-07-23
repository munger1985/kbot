"""Secret 引用校验适配器。"""

from __future__ import annotations

import hashlib
import os
from urllib.parse import urlparse

from aiops_agent.application.errors import dependency_unavailable, validation_failed
from aiops_agent.ports.secret_store import SecretReferenceMetadata


class ConfiguredSecretStore:
    """环境模式只检查变量是否存在；生产 Provider 留给专用适配器。"""

    def __init__(self, *, provider: str, allowed_schemes: tuple[str, ...]):
        self._provider = provider
        self._allowed_schemes = frozenset(allowed_schemes)

    async def validate_ref(self, reference: str) -> SecretReferenceMetadata:
        parsed = urlparse(reference)
        if (
            parsed.scheme not in self._allowed_schemes
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise validation_failed("SecretRef Provider 或格式不被允许")
        path = (parsed.netloc + parsed.path).strip("/")
        if not path or ".." in path.split("/"):
            raise validation_failed("SecretRef 路径无效")
        if self._provider == "environment":
            if parsed.scheme != "env":
                raise dependency_unavailable(
                    "当前部署未配置该 Secret Provider 适配器"
                )
            if path not in os.environ:
                raise validation_failed("SecretRef 指向的环境变量不存在")
        elif (
            self._provider == "vault" and parsed.scheme != "vault"
        ) or (
            self._provider == "secret_manager"
            and parsed.scheme != "secret-manager"
        ):
            raise validation_failed("SecretRef 与当前 Provider 不匹配")
        else:
            raise dependency_unavailable("Secret Provider 适配器尚未就绪")
        return SecretReferenceMetadata(
            provider=parsed.scheme,
            fingerprint=hashlib.sha256(
                reference.encode("utf-8")
            ).hexdigest()[:16],
        )
