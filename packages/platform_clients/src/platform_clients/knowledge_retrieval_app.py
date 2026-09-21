"""知识检索应用内部 Client。"""

from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext
from platform_core.security import build_scoped_internal_auth_headers


class KnowledgeRetrievalAppClientError(RuntimeError):
    """知识检索内部服务的稳定错误映射。"""

    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class KnowledgeRetrievalAppClient:
    _ROOT = "/internal/v1/knowledge-retrieval"
    _AGENTS = f"{_ROOT}/agents"

    def __init__(self, *, base_url: str, caller_service: str, audience: str, timeout_seconds: int = 120, session: aiohttp.ClientSession | None = None):
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def list_agents(self, *, domain_id: int, auth_context: AuthContext) -> list[dict[str, Any]]:
        return await self._json("GET", f"{self._AGENTS}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def create_agent(self, *, payload: dict[str, Any], auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("POST", self._AGENTS, payload=payload, auth_context=auth_context)

    async def get_agent(self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("GET", f"{self._AGENTS}/{agent_id}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def execution_spec(
        self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._AGENTS}/{agent_id}/execution-spec?"
            f"{urlencode({'domain_id': domain_id})}",
            auth_context=auth_context,
        )

    async def update_agent(self, *, agent_id: UUID, payload: dict[str, Any], auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("PATCH", f"{self._AGENTS}/{agent_id}", payload=payload, auth_context=auth_context)

    async def list_model_references(
        self, *, model_id: UUID, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        payload = await self._json(
            "GET",
            f"{self._AGENTS}/model-references/{model_id}",
            auth_context=auth_context,
        )
        return list(payload.get("references") or [])



    async def list_research_runs(
        self,
        *,
        domain_id: int,
        auth_context: AuthContext,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            self._query(f"{self._ROOT}/x-search/runs", domain_id=domain_id, status=status, limit=limit),
            auth_context=auth_context,
        )

    async def create_research_run(
        self,
        *,
        payload: dict[str, Any],
        auth_context: AuthContext,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._ROOT}/x-search/runs",
            payload=payload,
            auth_context=auth_context,
            extra_headers=self._idempotency_headers(idempotency_key),
        )

    async def get_research_run(self, *, run_id: UUID, domain_id: int, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._ROOT}/x-search/runs/{run_id}?{urlencode({'domain_id': domain_id})}",
            auth_context=auth_context,
        )

    async def list_research_events(self, *, run_id: UUID, domain_id: int, auth_context: AuthContext) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._ROOT}/x-search/runs/{run_id}/events?{urlencode({'domain_id': domain_id})}",
            auth_context=auth_context,
        )

    async def list_research_sources(self, *, run_id: UUID, domain_id: int, auth_context: AuthContext) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._ROOT}/x-search/runs/{run_id}/sources?{urlencode({'domain_id': domain_id})}",
            auth_context=auth_context,
        )

    async def delete_research_run(self, *, run_id: UUID, domain_id: int, auth_context: AuthContext) -> None:
        await self._json(
            "DELETE",
            f"{self._ROOT}/x-search/runs/{run_id}?{urlencode({'domain_id': domain_id})}",
            auth_context=auth_context,
        )











    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **build_scoped_internal_auth_headers(
                audience=self._audience, caller_service=self._caller_service,
                scopes=("knowledge_retrieval.manage",), context=auth_context,
            ),
            **(extra_headers or {}),
        }
        try:
            async with session.request(method, f"{self._base_url}{path}", headers=headers, json=payload) as response:
                body = await self._response_payload(response)
                if response.status >= 400:
                    raise self._error(response.status, body)
                return body
        except KnowledgeRetrievalAppClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise KnowledgeRetrievalAppClientError(
                status_code=503,
                code="KNOWLEDGE_RETRIEVAL_APP_UNAVAILABLE",
                message="知识检索服务暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()


    @staticmethod
    async def _response_payload(response: aiohttp.ClientResponse) -> Any:
        if response.status == 204:
            return None
        raw = await response.text()
        if not raw:
            return None
        try:
            return await response.json(content_type=None)
        except (aiohttp.ContentTypeError, ValueError):
            return raw

    @staticmethod
    def _error(status_code: int, body: Any) -> KnowledgeRetrievalAppClientError:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        code = (
            str(detail.get("code", "KNOWLEDGE_RETRIEVAL_APP_ERROR"))
            if isinstance(detail, dict)
            else "KNOWLEDGE_RETRIEVAL_APP_ERROR"
        )
        message = str(detail.get("message", detail)) if isinstance(detail, dict) else str(detail)
        return KnowledgeRetrievalAppClientError(status_code=status_code, code=code, message=message)

    @staticmethod
    def _idempotency_headers(idempotency_key: str | None) -> dict[str, str]:
        value = str(idempotency_key or "").strip()
        if not value:
            return {}
        return {"Idempotency-Key": value}

    @staticmethod
    def _query(path: str, **params: Any) -> str:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        return f"{path}?{query}" if query else path
