"""摄取端口只传中立值；实现由本 App 组合根固定。"""

from collections.abc import AsyncIterator
from typing import Protocol

from knowledge_algorithms.parsing.events import ParseEvent
from aiops_agent.domain.knowledge_ingestion import JobLease


class ParseComputationPort(Protocol):
    async def submit(self, lease: JobLease) -> str: ...
    async def events(self, lease: JobLease, *, after_sequence: int, limit: int) -> tuple[ParseEvent, ...]: ...
    async def acknowledge(self, lease: JobLease, *, sequence: int, event_digest: str) -> None: ...
    async def cancel(self, lease: JobLease) -> None: ...
    def artifact(self, lease: JobLease, handle_ref: str) -> AsyncIterator[bytes]: ...


class AuthorizedSourceReader:
    """每块重新验证当前 App 委托；取消或换代后不继续泄露源数据。"""
    def __init__(self, descriptor, store, authorize):
        self.descriptor, self.store, self.authorize = descriptor, store, authorize

    async def chunks(self):
        await self.authorize()
        iterator = self.store.stream(self.descriptor.source_handle)
        try:
            async for block in iterator:
                await self.authorize()
                yield block
        finally:
            await iterator.aclose()
