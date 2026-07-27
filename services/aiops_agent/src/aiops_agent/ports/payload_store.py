"""经验证监控原始正文的受控对象存储接口。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredMonitorPayload:
    uri: str
    content_hash: str
    byte_size: int


class MonitorPayloadStorePort(Protocol):
    async def store_verified(
        self, *, source_id: str, body: bytes, content_hash: str
    ) -> StoredMonitorPayload: ...

    async def delete(self, uri: str) -> None: ...
