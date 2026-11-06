"""完整资格范围的结构化检索端口；不可替换为相关性候选后排序。"""

from dataclasses import dataclass
from typing import Protocol

from ..contracts.identity import AppBackendBinding, AuthorizedScope, EvidenceKey, RetrievalSnapshot
from ..contracts.query import BackendCapabilities, QueryInput


@dataclass(frozen=True, slots=True)
class OrderedPage:
    snapshot: RetrievalSnapshot
    query_hash: str
    hits: tuple[EvidenceKey, ...]
    scope_exhausted: bool
    continuation: str | None


class OrderedSearchPort(Protocol):
    binding: AppBackendBinding
    capabilities: BackendCapabilities

    async def snapshot(self, scope: AuthorizedScope) -> RetrievalSnapshot: ...

    async def search_ordered(
        self, query: QueryInput, scope: AuthorizedScope, snapshot: RetrievalSnapshot,
    ) -> OrderedPage: ...
