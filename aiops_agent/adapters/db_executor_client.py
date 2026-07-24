"""带 Service Identity 的 DB Executor HTTP Client。"""

from __future__ import annotations

import aiohttp

from platform_core.contracts.aiops.executor import (
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)
from platform_core.security import build_scoped_internal_auth_headers


class DatabaseExecutorClientError(RuntimeError):
    pass


class DatabaseExecutorClient:
    def __init__(
        self,
        *,
        base_url: str,
        audience: str,
        caller_service: str,
        timeout_seconds: int,
        session: aiohttp.ClientSession,
    ):
        self._url = (
            base_url.rstrip("/")
            + "/internal/v1/db-executor/diagnostics"
        )
        self._audience = audience
        self._caller = caller_service
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def execute_diagnostic(
        self, request: ReadDiagnosticRequest, *, trace_id: str
    ) -> ReadDiagnosticResult:
        headers = build_scoped_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
            scopes=("db-executor.diagnostic",),
        )
        try:
            async with self._session.post(
                self._url,
                headers=headers,
                json=request.model_dump(mode="json"),
                timeout=self._timeout,
            ) as response:
                payload = await response.json()
                if response.status >= 400:
                    raise DatabaseExecutorClientError(
                        f"DB Executor 返回 HTTP {response.status}"
                    )
                return ReadDiagnosticResult.model_validate(payload)
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise DatabaseExecutorClientError(
                "DB Executor 本次不可用"
            ) from exc
