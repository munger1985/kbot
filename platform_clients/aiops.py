"""Main API 与 Agent Runtime 使用的两类窄 AIOps Client。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.contracts.aiops import (
    CreateOpsRunCommand,
    MonitorWebhookEnvelope,
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
        if_match: str | None = None,
    ) -> Any:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self._auth.headers(auth_context),
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if if_match:
            headers["If-Match"] = if_match
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
        if isinstance(body, dict) and "code" in body:
            code = str(body.get("code", "OPS_UPSTREAM_UNAVAILABLE"))
            message = str(body.get("detail", body.get("message", code)))
            retryable = bool(body.get("retryable", False))
        else:
            detail = (
                body.get("detail", body) if isinstance(body, dict) else body
            )
            if isinstance(detail, dict):
                code = str(
                    detail.get("code", "OPS_UPSTREAM_UNAVAILABLE")
                )
                message = str(
                    detail.get("detail", detail.get("message", code))
                )
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

    _CONFIG = f"{INTERNAL_API_V1}/aiops/config"

    @staticmethod
    def _list_path(
        path: str,
        *,
        status: str | None,
        cursor: str | None,
        limit: int,
    ) -> str:
        query = {"limit": str(limit)}
        if status:
            query["status"] = status
        if cursor:
            query["cursor"] = cursor
        return f"{path}?{urlencode(query)}"

    async def create_target(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_targets(
        self,
        *,
        status: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            self._list_path(
                f"{self._CONFIG}/targets",
                status=status,
                cursor=cursor,
                limit=limit,
            ),
            auth_context=auth_context,
        )

    async def get_target(
        self, target_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/targets/{target_id}",
            auth_context=auth_context,
        )

    async def patch_target(
        self,
        target_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{self._CONFIG}/targets/{target_id}",
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_target(
        self,
        target_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/{target_id}/{command}",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_agent_bindings(
        self, target_id: UUID, *, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/targets/{target_id}/agent-bindings",
            auth_context=auth_context,
        )

    async def create_agent_binding(
        self,
        target_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/{target_id}/agent-bindings",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def patch_agent_binding(
        self,
        target_id: UUID,
        binding_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            (
                f"{self._CONFIG}/targets/{target_id}"
                f"/agent-bindings/{binding_id}"
            ),
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_agent_binding(
        self,
        target_id: UUID,
        binding_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{self._CONFIG}/targets/{target_id}"
                f"/agent-bindings/{binding_id}/{command}"
            ),
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def create_monitor_source(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/monitor-sources",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_monitor_sources(
        self,
        *,
        status: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            self._list_path(
                f"{self._CONFIG}/monitor-sources",
                status=status,
                cursor=cursor,
                limit=limit,
            ),
            auth_context=auth_context,
        )

    async def get_monitor_source(
        self, source_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/monitor-sources/{source_id}",
            auth_context=auth_context,
        )

    async def patch_monitor_source(
        self,
        source_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{self._CONFIG}/monitor-sources/{source_id}",
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_monitor_source(
        self,
        source_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/monitor-sources/{source_id}/{command}",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def request_monitor_health_check(
        self,
        source_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/monitor-sources/{source_id}/health-checks",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def rotate_monitor_webhook_key(
        self,
        source_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/monitor-sources/{source_id}/webhook-key:rotate",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_monitor_bindings(
        self, target_id: UUID, *, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/targets/{target_id}/monitor-bindings",
            auth_context=auth_context,
        )

    async def create_monitor_binding(
        self,
        target_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/{target_id}/monitor-bindings",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def patch_monitor_binding(
        self,
        target_id: UUID,
        binding_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            (
                f"{self._CONFIG}/targets/{target_id}"
                f"/monitor-bindings/{binding_id}"
            ),
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_monitor_binding(
        self,
        target_id: UUID,
        binding_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{self._CONFIG}/targets/{target_id}"
                f"/monitor-bindings/{binding_id}/{command}"
            ),
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def create_policy(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/policies",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_policies(
        self,
        *,
        status: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            self._list_path(
                f"{self._CONFIG}/policies",
                status=status,
                cursor=cursor,
                limit=limit,
            ),
            auth_context=auth_context,
        )

    async def get_policy(
        self, policy_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/policies/{policy_id}",
            auth_context=auth_context,
        )

    async def command_policy(
        self,
        policy_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/policies/{policy_id}/{command}",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def create_inspection_plan(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/inspection-plans",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_inspection_plans(
        self,
        *,
        status: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            self._list_path(
                f"{self._CONFIG}/inspection-plans",
                status=status,
                cursor=cursor,
                limit=limit,
            ),
            auth_context=auth_context,
        )

    async def get_inspection_plan(
        self, plan_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/inspection-plans/{plan_id}",
            auth_context=auth_context,
        )

    async def patch_inspection_plan(
        self,
        plan_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{self._CONFIG}/inspection-plans/{plan_id}",
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_inspection_plan(
        self,
        plan_id: UUID,
        command: str,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/inspection-plans/{plan_id}/{command}",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_inspection_targets(
        self, plan_id: UUID, *, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/inspection-plans/{plan_id}/targets",
            auth_context=auth_context,
        )

    async def add_inspection_target(
        self,
        plan_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/inspection-plans/{plan_id}/targets",
            payload=payload,
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def patch_inspection_target(
        self,
        plan_id: UUID,
        plan_target_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            (
                f"{self._CONFIG}/inspection-plans/{plan_id}"
                f"/targets/{plan_target_id}"
            ),
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

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

    async def intake_monitor_event(
        self,
        envelope: MonitorWebhookEnvelope,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/intake/monitor-events",
            auth_context=auth_context,
            payload=envelope.model_dump(mode="json"),
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

    async def get_pending_input(
        self,
        run_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/runs/{run_id}/pending-input",
            auth_context=auth_context,
        )

    async def get_hitl_input(
        self,
        hitl_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/hitl/{hitl_id}",
            auth_context=auth_context,
        )

    async def respond_hitl(
        self,
        hitl_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/hitl/{hitl_id}/response",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def skip_hitl(
        self,
        hitl_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/hitl/{hitl_id}/skip",
            payload=payload,
            idempotency_key=idempotency_key,
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

    async def get_proposal(
        self,
        proposal_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/proposals/{proposal_id}",
            auth_context=auth_context,
        )

    async def reject_proposal(
        self,
        proposal_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/proposals/{proposal_id}/reject",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def approve_proposal(
        self,
        proposal_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/aiops/proposals/{proposal_id}"
                "/approve"
            ),
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def record_manual_result(
        self,
        proposal_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/aiops/proposals/{proposal_id}"
                "/manual-result"
            ),
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
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
