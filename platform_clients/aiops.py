"""Main API 与 Agent Runtime 使用的两类窄 AIOps Client。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.contracts.aiops import (
    CreateOpsRunCommand,
    OpsCommand,
    RootDelegationRequest,
)
from platform_core.security import (
    AuthContextJWTCodec,
    ServiceIdentityJWTCodec,
    build_scoped_internal_auth_headers,
)


@dataclass(frozen=True)
class AIOpsClientAuth:
    """Client 显式持有的调用身份和签名组件。"""

    caller_service: str
    audience: str
    scopes: tuple[str, ...]
    auth_context_codec: AuthContextJWTCodec
    service_identity_codec: ServiceIdentityJWTCodec

    def headers(self, context: AuthContext) -> dict[str, str]:
        return build_scoped_internal_auth_headers(
            audience=self.audience,
            caller_service=self.caller_service,
            scopes=self.scopes,
            context=context,
            auth_context_codec=self.auth_context_codec,
            service_identity_codec=self.service_identity_codec,
        )


class AIOpsClientError(RuntimeError):
    """AIOps Problem Details 的稳定 Client 映射。"""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retryable = retryable


class _BaseAIOpsClient:
    def __init__(
        self,
        *,
        base_url: str,
        auth: AIOpsClientAuth,
        timeout_seconds: int = 120,
        session: aiohttp.ClientSession | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session
        self._owns_session = session is None

    async def close(self) -> None:
        if (
            self._owns_session
            and self._session is not None
            and not self._session.closed
        ):
            await self._session.close()

    async def is_ready(self) -> bool:
        session = await self._get_session()
        try:
            async with session.get(f"{self._base_url}/ready") as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self._session

    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Any:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self._auth.headers(auth_context),
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        session = await self._get_session()
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                json=payload,
            ) as response:
                body = await self._response_payload(response)
                if response.status >= 400:
                    self._raise_error(response.status, body)
                return body
        except AIOpsClientError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AIOpsClientError(
                status_code=503,
                code="OPS_UPSTREAM_UNAVAILABLE",
                message="AIOps 服务暂时不可用",
                retryable=True,
            ) from exc

    @staticmethod
    async def _response_payload(response: aiohttp.ClientResponse) -> Any:
        if response.status == 204:
            return None
        try:
            return await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            return {"detail": await response.text()}

    @staticmethod
    def _raise_error(status_code: int, body: Any) -> None:
        detail = body.get("detail", body) if isinstance(body, dict) else body
        if isinstance(detail, dict):
            code = str(detail.get("code", "OPS_UPSTREAM_UNAVAILABLE"))
            message = str(detail.get("detail", detail.get("message", code)))
            retryable = bool(detail.get("retryable", False))
        else:
            code = "OPS_UPSTREAM_UNAVAILABLE"
            message = str(detail)
            retryable = status_code in {429, 502, 503, 504}
        raise AIOpsClientError(
            status_code=status_code,
            code=code,
            message=message,
            retryable=retryable,
        )


class AIOpsManagementClient(_BaseAIOpsClient):
    """Main API 的管理、用户命令和 Direct Run Client。"""

    async def create_run(
        self,
        command: CreateOpsRunCommand,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/runs",
            auth_context=auth_context,
            payload=command.model_dump(mode="json"),
            idempotency_key=command.idempotency_key,
        )

    async def get_run(
        self,
        run_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/runs/{run_id}",
            auth_context=auth_context,
        )

    async def list_run_events(
        self,
        run_id: UUID,
        *,
        after_sequence: int,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/runs/{run_id}/events"
                f"?after={after_sequence}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def command(
        self,
        command: OpsCommand,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/runs/{command.ops_run_id}/commands",
            auth_context=auth_context,
            payload=command.model_dump(mode="json"),
            idempotency_key=command.idempotency_key,
        )


class AIOpsDelegationClient(_BaseAIOpsClient):
    """Agent Runtime 只能使用的 Root Delegation Client。"""

    async def create_delegation(
        self,
        request: RootDelegationRequest,
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/delegations",
            auth_context=auth_context,
            payload=request.model_dump(mode="json"),
            idempotency_key=idempotency_key,
        )

    async def list_events(
        self,
        delegation_id: UUID,
        *,
        after_sequence: int,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/delegations/{delegation_id}/events"
                f"?after={after_sequence}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def get_result(
        self,
        delegation_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/delegations/{delegation_id}/result",
            auth_context=auth_context,
        )

    async def cancel(
        self,
        delegation_id: UUID,
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/delegations/{delegation_id}/cancel",
            auth_context=auth_context,
            payload={},
            idempotency_key=idempotency_key,
        )
