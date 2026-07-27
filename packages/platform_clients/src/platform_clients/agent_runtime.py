"""Main API 调用 Agent Runtime 的窄内部 Client。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class AgentRuntimeClientError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class AgentRuntimeClient:
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

    async def is_ready(self) -> bool:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(
            timeout=self._timeout
        )
        try:
            async with session.get(f"{self._base_url}/readyz") as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False
        finally:
            if owns_session:
                await session.close()

    async def create_agent(
        self, *, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/agents",
            payload=payload,
            auth_context=auth_context,
        )

    async def list_agents(
        self, *, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/agents",
            auth_context=auth_context,
        )

    async def get_agent(
        self, *, agent_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/agents/{agent_id}",
            auth_context=auth_context,
        )

    async def update_agent(
        self,
        *,
        agent_id: UUID,
        payload: dict[str, Any],
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{INTERNAL_API_V1}/agents/{agent_id}",
            payload=payload,
            auth_context=auth_context,
        )

    async def list_data_profiles(
        self, *, auth_context: AuthContext
    ) -> Any:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/data/profiles",
            auth_context=auth_context,
        )

    async def create_run(
        self,
        *,
        payload: dict[str, Any],
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/runs",
            payload=payload,
            auth_context=auth_context,
            extra_headers={"Idempotency-Key": idempotency_key},
        )

    async def create_conversation(
        self, *, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/conversations",
            payload=payload,
            auth_context=auth_context,
        )

    async def list_conversations(
        self, *, limit: int, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/conversations?limit={limit}",
            auth_context=auth_context,
        )

    async def get_conversation(
        self, *, conversation_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/conversations/{conversation_id}",
            auth_context=auth_context,
        )

    async def update_conversation(
        self,
        *,
        conversation_id: UUID,
        payload: dict[str, Any],
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{INTERNAL_API_V1}/conversations/{conversation_id}",
            payload=payload,
            auth_context=auth_context,
        )

    async def delete_conversation(
        self,
        *,
        conversation_id: UUID,
        expected_row_version: int,
        auth_context: AuthContext,
    ) -> None:
        await self._json(
            "DELETE",
            (
                f"{INTERNAL_API_V1}/conversations/{conversation_id}"
                f"?expected_row_version={expected_row_version}"
            ),
            auth_context=auth_context,
        )

    async def create_conversation_turn(
        self,
        *,
        conversation_id: UUID,
        payload: dict[str, Any],
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/conversations/"
                f"{conversation_id}/turns"
            ),
            payload=payload,
            auth_context=auth_context,
            extra_headers={"Idempotency-Key": idempotency_key},
        )

    async def list_conversation_turns(
        self,
        *,
        conversation_id: UUID,
        after: int,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/conversations/{conversation_id}/turns"
                f"?after={after}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def list_turn_trace(
        self,
        *,
        conversation_id: UUID,
        turn_id: UUID,
        after: int,
        limit: int,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/conversations/{conversation_id}/turns/"
                f"{turn_id}/trace?after={after}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def list_memories(
        self,
        *,
        agent_id: UUID,
        limit: int,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/memories"
                f"?agent_id={agent_id}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def forget_memory(
        self, *, memory_id: UUID, auth_context: AuthContext
    ) -> None:
        await self._json(
            "DELETE",
            f"{INTERNAL_API_V1}/memories/{memory_id}",
            auth_context=auth_context,
        )

    async def get_run(
        self, *, run_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/runs/{run_id}",
            auth_context=auth_context,
        )

    async def list_debug_runs(
        self, *, limit: int, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/runs/development/recent?limit={limit}",
            auth_context=auth_context,
        )

    async def get_debug_run(
        self, *, run_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/runs/{run_id}/development",
            auth_context=auth_context,
        )

    async def get_result(
        self, *, run_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/runs/{run_id}/result",
            auth_context=auth_context,
        )

    async def list_events(
        self,
        *,
        run_id: UUID,
        after_sequence: int,
        limit: int,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/runs/{run_id}/events"
                f"?after={after_sequence}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def cancel_run(
        self,
        *,
        run_id: UUID,
        expected_row_version: int,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/runs/{run_id}/cancel",
            payload={"expected_row_version": expected_row_version},
            auth_context=auth_context,
            extra_headers={"Idempotency-Key": idempotency_key},
        )

    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(
            timeout=self._timeout
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **build_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
                context=auth_context,
            ),
            **(extra_headers or {}),
        }
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=payload,
            ) as response:
                body = await self._response_payload(response)
                if response.status >= 400:
                    detail = (
                        body.get("detail", body)
                        if isinstance(body, dict)
                        else body
                    )
                    code = (
                        str(detail.get("code", "AGENT_RUNTIME_ERROR"))
                        if isinstance(detail, dict)
                        else "AGENT_RUNTIME_ERROR"
                    )
                    message = (
                        str(detail.get("message", detail))
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    raise AgentRuntimeClientError(
                        status_code=response.status,
                        code=code,
                        message=message,
                    )
                return body
        except AgentRuntimeClientError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AgentRuntimeClientError(
                status_code=503,
                code="AGENT_RUNTIME_UNAVAILABLE",
                message="Agent Runtime 暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()

    @staticmethod
    async def _response_payload(response: aiohttp.ClientResponse) -> Any:
        if response.status == 204:
            return None
        try:
            return await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            return {"message": await response.text()}
