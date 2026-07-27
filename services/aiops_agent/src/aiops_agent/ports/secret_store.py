"""Secret 引用元数据校验 Port。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SecretReferenceMetadata:
    provider: str
    fingerprint: str


@dataclass(frozen=True, repr=False)
class ResolvedSecret:
    """仅在 Adapter 调用期存在，禁止日志和 Artifact 序列化。"""

    values: dict[str, str]
    fingerprint: str


class SecretStorePort(Protocol):
    async def validate_ref(self, reference: str) -> SecretReferenceMetadata:
        """只验证元数据与可访问性，不读取 Secret Value。"""

    async def resolve(self, reference: str) -> ResolvedSecret:
        """在调用 Provider 前解析短期凭据。"""
