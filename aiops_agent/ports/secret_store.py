"""Secret 引用元数据校验 Port。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SecretReferenceMetadata:
    provider: str
    fingerprint: str


class SecretStorePort(Protocol):
    async def validate_ref(self, reference: str) -> SecretReferenceMetadata:
        """只验证元数据与可访问性，不读取 Secret Value。"""
