"""DB Executor 访问 AIOps 控制面的反向 Claim 与状态回调 Client。"""

from __future__ import annotations

from uuid import UUID

import aiohttp

from platform_core.contracts.aiops.executor import (
    CredentialIssueRequest,
    CredentialIssueResponse,
    ExecutionStatusEvent,
    MutationClaimReceipt,
    MutationClaimRequest,
)
from platform_core.contracts.aiops.internal import EventReceipt
from platform_core.security import (
    build_scoped_internal_auth_headers,
    create_service_auth_context,
)


class AIOpsExecutionClientError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class AIOpsExecutionClient:
    def __init__(
        self,
        *,
        base_url: str,
        audience: str,
        caller_service: str,
        timeout_seconds: int,
        session: aiohttp.ClientSession,
    ):
        self._base_url = base_url.rstrip("/")
        self._audience = audience
        self._caller = caller_service
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def claim_execution(
        self,
        execution_id: UUID,
        request: MutationClaimRequest,
        *,
        trace_id: str,
    ) -> MutationClaimReceipt:
        return await self._post(
            path=f"/internal/v1/aiops/executions/{execution_id}/claim",
            payload=request.model_dump(mode="json"),
            scope="aiops.execution.claim",
            trace_id=trace_id,
            response_type=MutationClaimReceipt,
        )

    async def publish_event(
        self,
        event: ExecutionStatusEvent,
        *,
        trace_id: str,
    ) -> EventReceipt:
        return await self._post(
            path="/internal/v1/aiops/executor-events",
            payload=event.model_dump(mode="json"),
            scope="aiops.execution.callback",
            trace_id=trace_id,
            response_type=EventReceipt,
        )

    async def issue_credential(self, grant: str, *, trace_id: str) -> CredentialIssueResponse:
        return await self._post(path="/internal/v1/aiops/credentials:issue", payload=CredentialIssueRequest(grant=grant).model_dump(mode="json"), scope="aiops.credentials.issue", trace_id=trace_id, response_type=CredentialIssueResponse)

    async def _post(
        self,
        *,
        path: str,
        payload: dict,
        scope: str,
        trace_id: str,
        response_type,
    ):
        headers = build_scoped_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller,
            scopes=(scope,),
            context=create_service_auth_context(
                caller_service=self._caller,
                trace_id=trace_id,
            ),
        )
        try:
            async with self._session.post(
                self._base_url + path,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            ) as response:
                body = await response.json()
                if response.status >= 400:
                    detail = body.get("detail", body)
                    code = (
                        detail.get("code")
                        if isinstance(detail, dict)
                        else None
                    )
                    raise AIOpsExecutionClientError(
                        code or f"AIOPS_HTTP_{response.status}"
                    )
                return response_type.model_validate(body)
        except AIOpsExecutionClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise AIOpsExecutionClientError(
                "AIOPS_CONTROL_PLANE_UNAVAILABLE"
            ) from exc
