"""KC 内部检索契约的认证 HTTP 客户端。"""
from typing import Any, Sequence

import aiohttp

from platform_core.config.settings import get_app_config, get_knowledge_core_config
from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class KnowledgeCoreClientError(RuntimeError):
    pass


class KnowledgeCoreClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int = 120,
        session: aiohttp.ClientSession | None = None,
        caller_service: str | None = None,
        audience: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session
        self._caller_service = caller_service or get_app_config().service_name
        self._audience = audience or get_knowledge_core_config().service_name

    async def discover(self, *, query: str, collection_ids: Sequence[int], domain_id: int | None = None, agent_id: int | None = None, query_vectors: dict[int, Sequence[float]] | None = None, max_security_level: int = 3, per_collection_limit: int = 20, auth_context: AuthContext | None = None) -> dict[str, Any]:
        return await self._request("POST", f"{INTERNAL_API_V1}/knowledge/discovery/search", {
            "query": query, "collection_ids": list(collection_ids), "domain_id": domain_id, "agent_id": agent_id, "query_vectors": query_vectors,
            "max_security_level": max_security_level, "per_collection_limit": per_collection_limit,
        }, auth_context=auth_context)

    async def retrieve_evidence(self, *, query: str, candidates: Sequence[dict[str, Any]], domain_id: int | None = None, agent_id: int | None = None, query_vectors: dict[int, Sequence[float]] | None = None, max_security_level: int = 3, max_evidence: int = 12, context_limit: int = 4, auth_context: AuthContext | None = None) -> dict[str, Any]:
        return await self._request("POST", f"{INTERNAL_API_V1}/knowledge/retrieval/evidence", {
            "query": query, "candidates": list(candidates), "domain_id": domain_id, "agent_id": agent_id, "query_vectors": query_vectors,
            "max_security_level": max_security_level, "max_evidence": max_evidence, "context_limit": context_limit,
        }, auth_context=auth_context)

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        *,
        auth_context: AuthContext | None,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            **build_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
                context=auth_context,
            ),
        }
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
