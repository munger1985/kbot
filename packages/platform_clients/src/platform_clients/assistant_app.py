"""智能工作台内部 Client。"""

from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext
from platform_core.security import build_scoped_internal_auth_headers


class AssistantAppClientError(RuntimeError):
    """智能工作台内部服务的稳定错误映射。"""

    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class AssistantAppClient:
    _BASE = "/internal/v1/assistant/agents"

    def __init__(self, *, base_url: str, caller_service: str, audience: str, timeout_seconds: int = 120, session: aiohttp.ClientSession | None = None):
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def list_agents(self, *, domain_id: int, auth_context: AuthContext) -> list[dict[str, Any]]:
        return await self._json("GET", f"{self._BASE}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def create_agent(self, *, payload: dict[str, Any], auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("POST", self._BASE, payload=payload, auth_context=auth_context)

    async def get_agent(self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("GET", f"{self._BASE}/{agent_id}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def update_agent(self, *, agent_id: UUID, payload: dict[str, Any], auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("PATCH", f"{self._BASE}/{agent_id}", payload=payload, auth_context=auth_context)

    async def _json(self, method: str, path: str, *, auth_context: AuthContext, payload: dict[str, Any] | None = None):
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {
            "Accept": "application/json", "Content-Type": "application/json",
            **build_scoped_internal_auth_headers(
                audience=self._audience, caller_service=self._caller_service,
                scopes=("assistant.manage",), context=auth_context,
            ),
        }
        try:
            async with session.request(method, f"{self._base_url}{path}", headers=headers, json=payload) as response:
                body = await response.json()
                if response.status >= 400:
                    detail = body.get("detail", body) if isinstance(body, dict) else body
                    code = str(detail.get("code", "ASSISTANT_APP_ERROR")) if isinstance(detail, dict) else "ASSISTANT_APP_ERROR"
                    message = str(detail.get("message", detail)) if isinstance(detail, dict) else str(detail)
                    raise AssistantAppClientError(status_code=response.status, code=code, message=message)
                return body
        except AssistantAppClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise AssistantAppClientError(status_code=503, code="ASSISTANT_APP_UNAVAILABLE", message="智能工作台服务暂时不可用") from exc
        finally:
            if owns_session:
                await session.close()
