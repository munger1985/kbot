"""知识检索应用内部 Client。"""

from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext
from platform_core.security import build_scoped_internal_auth_headers


class KnowledgeRetrievalAppClientError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class KnowledgeRetrievalAppClient:
    _BASE = "/internal/v1/knowledge-retrieval/agents"

    def __init__(
        self,
        *,
        base_url: str,
        caller_service: str,
        audience: str,
        timeout_seconds: int = 120,
        session: aiohttp.ClientSession | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def list_agents(self, *, domain_id: int, auth_context: AuthContext):
        query = urlencode({"domain_id": str(domain_id)})
        return await self._json(
            "GET", f"{self._BASE}?{query}", auth_context=auth_context
        )

    async def create_agent(
        self, *, payload: dict[str, Any], auth_context: AuthContext
    ):
        return await self._json(
            "POST", self._BASE, payload=payload, auth_context=auth_context
        )

    async def get_agent(
        self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext
    ):
        query = urlencode({"domain_id": str(domain_id)})
        return await self._json(
            "GET",
            f"{self._BASE}/{agent_id}?{query}",
            auth_context=auth_context,
        )

    async def update_agent(
        self,
        *,
        agent_id: UUID,
        payload: dict[str, Any],
        auth_context: AuthContext,
    ):
        return await self._json(
            "PATCH",
            f"{self._BASE}/{agent_id}",
            payload=payload,
            auth_context=auth_context,
        )

    async def execution_spec(
        self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext
    ):
        query = urlencode({"domain_id": str(domain_id)})
        return await self._json(
            "GET",
            f"{self._BASE}/{agent_id}/execution-spec?{query}",
            auth_context=auth_context,
        )

    async def list_model_references(
        self, *, model_id: UUID, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        payload = await self._json(
            "GET",
            f"{self._BASE}/model-references/{model_id}",
            auth_context=auth_context,
        )
        return list(payload.get("references") or [])

    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
    ):
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **build_scoped_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
                scopes=("knowledge_retrieval.manage",),
                context=auth_context,
            ),
        }
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=payload,
            ) as response:
                body = await response.json()
                if response.status >= 400:
                    detail = body.get("detail", body) if isinstance(body, dict) else body
                    code = (
                        str(detail.get("code", "KNOWLEDGE_RETRIEVAL_APP_ERROR"))
                        if isinstance(detail, dict)
                        else "KNOWLEDGE_RETRIEVAL_APP_ERROR"
                    )
                    message = (
                        str(detail.get("message", detail))
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    raise KnowledgeRetrievalAppClientError(
                        status_code=response.status,
                        code=code,
                        message=message,
                    )
                return body
        except KnowledgeRetrievalAppClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise KnowledgeRetrievalAppClientError(
                status_code=503,
                code="KNOWLEDGE_RETRIEVAL_APP_UNAVAILABLE",
                message="知识检索应用服务暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()
