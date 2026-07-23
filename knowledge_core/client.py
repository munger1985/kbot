"""KC 内部检索契约的认证 HTTP 客户端。"""
from typing import Any, Sequence

import aiohttp

from platform_core.contracts import INTERNAL_API_V1
from platform_core.platform.security import INTERNAL_TOKEN_HEADER, get_internal_token


class KnowledgeCoreClientError(RuntimeError):
    pass


class KnowledgeCoreClient:
    def __init__(self, *, base_url: str, timeout_seconds: int = 120, session: aiohttp.ClientSession | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def discover(self, *, query: str, collection_ids: Sequence[int], domain_id: int | None = None, agent_id: int | None = None, query_vectors: dict[int, Sequence[float]] | None = None, max_security_level: int = 3, per_collection_limit: int = 20) -> dict[str, Any]:
        return await self._request("POST", f"{INTERNAL_API_V1}/knowledge/discovery/search", {
            "query": query, "collection_ids": list(collection_ids), "domain_id": domain_id, "agent_id": agent_id, "query_vectors": query_vectors,
            "max_security_level": max_security_level, "per_collection_limit": per_collection_limit,
        })

    async def retrieve_evidence(self, *, query: str, candidates: Sequence[dict[str, Any]], domain_id: int | None = None, agent_id: int | None = None, query_vectors: dict[int, Sequence[float]] | None = None, max_security_level: int = 3, max_evidence: int = 12, context_limit: int = 4) -> dict[str, Any]:
        return await self._request("POST", f"{INTERNAL_API_V1}/knowledge/retrieval/evidence", {
            "query": query, "candidates": list(candidates), "domain_id": domain_id, "agent_id": agent_id, "query_vectors": query_vectors,
            "max_security_level": max_security_level, "max_evidence": max_evidence, "context_limit": context_limit,
        })

    async def _request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json", INTERNAL_TOKEN_HEADER: get_internal_token()}
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        try:
            async with session.request(method, f"{self._base_url}{path}", json=payload, headers=headers) as response:
                if response.status >= 400:
                    raise KnowledgeCoreClientError(f"KC HTTP {response.status}: {await response.text()}")
                return await response.json()
        finally:
            if owns_session:
                await session.close()
