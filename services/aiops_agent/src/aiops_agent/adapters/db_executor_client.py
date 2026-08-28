"""带 Service Identity 的 DB Executor HTTP Client。"""

from __future__ import annotations

import aiohttp

from platform_core.contracts.aiops.executor import (
    DynamicReadDiagnosticRequest,
    ExecutionResultRef,
    MutationExecutionRequest,
    ReadDiagnosticRequest,
    ReadDiagnosticResult,
)
from platform_core.security import (
    build_scoped_internal_auth_headers,
    create_service_auth_context,
)


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
        self._diagnostic_url = (
            base_url.rstrip("/")
            + "/internal/v1/db-executor/diagnostics"
        )
        self._mutation_url = (
            base_url.rstrip("/")
            + "/internal/v1/db-executor/executions"
        )
        self._dynamic_diagnostic_url = (
            base_url.rstrip("/")
            + "/internal/v1/db-executor/dynamic-diagnostics"
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
                self._diagnostic_url,
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

    async def request_execution(
        self, request: MutationExecutionRequest, *, trace_id: str
    ) -> ExecutionResultRef:
        """向隔离 Executor 投递一次执行通知；授权由 Executor 反向 Claim。"""
        headers = build_scoped_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
            scopes=("db-executor.mutation",),
            context=create_service_auth_context(
                caller_service=self._caller,
                trace_id=trace_id,
            ),
        )
        try:
            async with self._session.post(
                self._mutation_url,
                headers=headers,
                json=request.model_dump(mode="json"),
                timeout=self._timeout,
            ) as response:
                payload = await response.json()
                if response.status >= 400:
                    raise DatabaseExecutorClientError(
                        f"DB Executor 返回 HTTP {response.status}"
                    )
                return ExecutionResultRef.model_validate(payload)
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise DatabaseExecutorClientError(
                "DB Executor 变更入口本次不可用"
            ) from exc

    async def execute_dynamic_diagnostic(
        self,
        request: DynamicReadDiagnosticRequest,
        *,
        trace_id: str,
    ) -> ReadDiagnosticResult:
        """调用与固定目录隔离的动态只读查询入口。"""
        del trace_id
        headers = build_scoped_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
            scopes=("db-executor.diagnostic",),
        )
        try:
            async with self._session.post(
                self._dynamic_diagnostic_url,
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
                "DB Executor 动态诊断入口本次不可用"
            ) from exc
