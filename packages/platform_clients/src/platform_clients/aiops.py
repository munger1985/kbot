"""Main API 与 Agent Runtime 使用的两类窄 AIOps Client。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.parse import quote
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.contracts.aiops import (
    CreateOpsRunCommand,
    SignalEventEnvelope,
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
                # Client 可能复用进程级 Session；每个依赖仍必须使用自己的
                # 超时，不能被其他服务更长的 Session 默认超时覆盖。
                timeout=self._timeout,
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

    async def _upload(
        self,
        path: str,
        *,
        auth_context: AuthContext,
        file_name: str,
        media_type: str,
        body,
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json",
            "Content-Type": media_type,
            "X-File-Name": quote(file_name, safe=""),
            **self._auth.headers(auth_context),
        }
        session = await self._get_session()
        try:
            async with session.post(
                f"{self._base_url}{path}",
                headers=headers,
                data=body,
                timeout=self._timeout,
            ) as response:
                payload = await self._response_payload(response)
                if response.status >= 400:
                    self._raise_error(response.status, payload)
                return payload
        except AIOpsClientError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AIOpsClientError(
                status_code=503,
                code="OPS_UPSTREAM_UNAVAILABLE",
                message="AIOps 上传服务暂时不可用",
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

    async def list_notification_subscriptions(
        self, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/notification-subscriptions",
            auth_context=auth_context,
        )

    async def upsert_notification_subscription(
        self,
        target_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PUT",
            f"{self._CONFIG}/notification-subscriptions/targets/{target_id}",
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def disable_notification_subscription(
        self,
        target_id: UUID,
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> None:
        await self._json(
            "DELETE",
            f"{self._CONFIG}/notification-subscriptions/targets/{target_id}",
            if_match=if_match,
            auth_context=auth_context,
        )

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

    async def test_target_connection(
        self,
        payload: dict[str, Any],
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/test-connection",
            payload=payload,
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

    async def request_target_connectivity_check(
        self,
        target_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/{target_id}/connectivity-checks",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def rotate_target_credential(self, target_id: UUID, kind: str, payload: dict[str, Any], *, if_match: str, idempotency_key: str, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("POST", f"{self._CONFIG}/targets/{target_id}/{kind}-credential:rotate", payload=payload, if_match=if_match, idempotency_key=idempotency_key, auth_context=auth_context)

    async def remove_execution_credential(self, target_id: UUID, *, if_match: str, idempotency_key: str, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("POST", f"{self._CONFIG}/targets/{target_id}/execution-credential:remove", payload={}, if_match=if_match, idempotency_key=idempotency_key, auth_context=auth_context)

    async def remove_diagnostic_credential(self, target_id: UUID, *, if_match: str, idempotency_key: str, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("POST", f"{self._CONFIG}/targets/{target_id}/diagnostic-credential:remove", payload={}, if_match=if_match, idempotency_key=idempotency_key, auth_context=auth_context)

    async def delete_target(self, target_id: UUID, *, if_match: str, idempotency_key: str, auth_context: AuthContext) -> None:
        await self._json("DELETE", f"{self._CONFIG}/targets/{target_id}", if_match=if_match, idempotency_key=idempotency_key, auth_context=auth_context)

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

    async def create_diagnostic_source(
        self,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/diagnostic-sources",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def test_diagnostic_source_connection(
        self,
        payload: dict[str, Any],
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/diagnostic-sources/test-connection",
            payload=payload,
            auth_context=auth_context,
        )

    async def list_diagnostic_sources(
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
                f"{self._CONFIG}/diagnostic-sources",
                status=status,
                cursor=cursor,
                limit=limit,
            ),
            auth_context=auth_context,
        )

    async def get_diagnostic_source(
        self, source_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/diagnostic-sources/{source_id}",
            auth_context=auth_context,
        )

    async def patch_diagnostic_source(
        self,
        source_id: UUID,
        payload: dict[str, Any],
        *,
        if_match: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            f"{self._CONFIG}/diagnostic-sources/{source_id}",
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def delete_diagnostic_source(
        self,
        source_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> None:
        await self._json(
            "DELETE",
            f"{self._CONFIG}/diagnostic-sources/{source_id}",
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def command_diagnostic_source(
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
            f"{self._CONFIG}/diagnostic-sources/{source_id}/{command}",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def request_diagnostic_source_connectivity_check(
        self,
        source_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/diagnostic-sources/{source_id}/connectivity-checks",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def rotate_diagnostic_source_webhook_key(
        self,
        source_id: UUID,
        *,
        if_match: str,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/diagnostic-sources/{source_id}/webhook-key:rotate",
            payload={},
            if_match=if_match,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def list_source_bindings(
        self, target_id: UUID, *, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            f"{self._CONFIG}/targets/{target_id}/source-bindings",
            auth_context=auth_context,
        )

    async def create_source_binding(
        self,
        target_id: UUID,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{self._CONFIG}/targets/{target_id}/source-bindings",
            payload=payload,
            idempotency_key=idempotency_key,
            auth_context=auth_context,
        )

    async def patch_source_binding(
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
                f"/source-bindings/{binding_id}"
            ),
            payload=payload,
            if_match=if_match,
            auth_context=auth_context,
        )

    async def command_source_binding(
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
                f"/source-bindings/{binding_id}/{command}"
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

    async def intake_signal_event(
        self,
        envelope: SignalEventEnvelope,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/intake/signal-events",
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

    async def list_runs(self, *, target_id: UUID | None, status: str | None,
                        cursor: str | None, limit: int,
                        auth_context: AuthContext) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if target_id is not None:
            query["target_id"] = str(target_id)
        if status is not None:
            query["status"] = status
        if cursor is not None:
            query["cursor"] = cursor
        return await self._json("GET", f"{INTERNAL_API_V1}/aiops/runs?{urlencode(query)}",
                                auth_context=auth_context)

    async def list_situations(self, *, target_id: UUID | None, status: str | None,
                              severity: str | None, cursor: str | None,
                              limit: int, auth_context: AuthContext) -> dict[str, Any]:
        query = {"limit": str(limit)}
        for key, value in (("target_id", target_id), ("status", status),
                           ("severity", severity), ("cursor", cursor)):
            if value is not None:
                query[key] = str(value)
        return await self._json("GET", f"{INTERNAL_API_V1}/aiops/situations?{urlencode(query)}",
                                auth_context=auth_context)

    async def get_situation(self, situation_id: UUID, *,
                            auth_context: AuthContext) -> dict[str, Any]:
        return await self._json("GET", f"{INTERNAL_API_V1}/aiops/situations/{situation_id}",
                                auth_context=auth_context)

    async def get_run_result(
        self,
        run_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/runs/{run_id}/result",
            auth_context=auth_context,
        )

    async def list_proposals(self, *, target_id: UUID | None,
                             status: str | None, cursor: str | None,
                             limit: int, auth_context: AuthContext) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if target_id is not None:
            query["target_id"] = str(target_id)
        if status is not None:
            query["status"] = status
        if cursor is not None:
            query["cursor"] = cursor
        return await self._json(
            "GET", f"{INTERNAL_API_V1}/aiops/proposals?{urlencode(query)}",
            auth_context=auth_context,
        )

    async def get_report(
        self,
        report_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/reports/{report_id}",
            auth_context=auth_context,
        )

    async def list_reports(
        self,
        *,
        target_id: UUID | None,
        report_type: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if target_id is not None:
            query["target_id"] = str(target_id)
        if report_type is not None:
            query["report_type"] = report_type
        if cursor is not None:
            query["cursor"] = cursor
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/reports?"
                f"{urlencode(query)}"
            ),
            auth_context=auth_context,
        )

    async def list_report_versions(
        self,
        report_id: UUID,
        *,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if cursor is not None:
            query["cursor"] = cursor
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/reports/{report_id}/versions?"
                f"{urlencode(query)}"
            ),
            auth_context=auth_context,
        )

    async def list_inspection_fires(
        self,
        *,
        plan_id: UUID | None,
        status: str | None,
        cursor: str | None,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        query = {"limit": str(limit)}
        if plan_id is not None:
            query["plan_id"] = str(plan_id)
        if status is not None:
            query["status"] = status
        if cursor is not None:
            query["cursor"] = cursor
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/inspection-fires?"
                f"{urlencode(query)}"
            ),
            auth_context=auth_context,
        )

    async def get_inspection_fire(
        self,
        fire_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/inspection-fires/{fire_id}",
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


    async def list_private_agents(self, *, auth_context: AuthContext):
        return await self._json(
            "GET", f"{INTERNAL_API_V1}/aiops/agents",
            auth_context=auth_context,
        )

    async def get_private_agent(
        self, agent_id: UUID, *, auth_context: AuthContext
    ):
        return await self._json(
            "GET", f"{INTERNAL_API_V1}/aiops/agents/{agent_id}",
            auth_context=auth_context,
        )

    async def create_private_agent(
        self, payload: dict[str, Any], *, auth_context: AuthContext
    ):
        return await self._json(
            "POST", f"{INTERNAL_API_V1}/aiops/agents",
            payload=payload, auth_context=auth_context,
        )

    async def update_private_agent(
        self, agent_id: UUID, payload: dict[str, Any], *, auth_context: AuthContext
    ):
        return await self._json(
            "PATCH", f"{INTERNAL_API_V1}/aiops/agents/{agent_id}",
            payload=payload, auth_context=auth_context,
        )

    async def list_model_references(
        self, *, model_id: UUID, auth_context: AuthContext
    ) -> list[dict[str, Any]]:
        payload = await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/agents/model-references/{model_id}",
            auth_context=auth_context,
        )
        return list(payload.get("references") or [])

    async def list_private_agent_grants(self, *, auth_context: AuthContext):
        return await self._json(
            "GET", f"{INTERNAL_API_V1}/aiops/agents/grants/list",
            auth_context=auth_context,
        )

    async def upsert_private_agent_grant(
        self, payload: dict[str, Any], *, auth_context: AuthContext
    ):
        return await self._json(
            "PUT", f"{INTERNAL_API_V1}/aiops/agents/grants",
            payload=payload, auth_context=auth_context,
        )

    async def update_private_agent_grant(
        self, grant_id: UUID, payload: dict[str, Any], *, auth_context: AuthContext
    ):
        return await self._json(
            "PATCH", f"{INTERNAL_API_V1}/aiops/agents/grants/{grant_id}",
            payload=payload, auth_context=auth_context,
        )

    async def authorize_private_agent(
        self, payload: dict[str, Any], *, auth_context: AuthContext
    ):
        return await self._json(
            "POST", f"{INTERNAL_API_V1}/aiops/agents:authorize",
            payload=payload, auth_context=auth_context,
        )

    async def start_conversation(
        self, payload: dict[str, Any], *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/conversations",
            payload=payload,
            auth_context=auth_context,
        )

    async def upload_conversation_input(
        self,
        *,
        file_name: str,
        media_type: str,
        body,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._upload(
            f"{INTERNAL_API_V1}/aiops/conversations/uploads",
            auth_context=auth_context,
            file_name=file_name,
            media_type=media_type,
            body=body,
        )

    async def list_conversations(
        self,
        *,
        agent_id: UUID | None,
        limit: int,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        path = f"{INTERNAL_API_V1}/aiops/conversations?limit={limit}"
        if agent_id is not None:
            path += f"&agent_id={agent_id}"
        return await self._json("GET", path, auth_context=auth_context)

    async def get_conversation(
        self, conversation_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}",
            auth_context=auth_context,
        )

    async def archive_conversation(
        self, conversation_id: UUID, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "DELETE",
            f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}",
            auth_context=auth_context,
        )

    async def create_conversation_turn(
        self,
        conversation_id: UUID,
        payload: dict[str, Any],
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}/turns",
            payload=payload,
            auth_context=auth_context,
        )

    async def list_conversation_turns(
        self,
        conversation_id: UUID,
        *,
        after_turn_no: int,
        limit: int,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}/turns"
                f"?after_turn_no={after_turn_no}&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def get_conversation_turn(
        self,
        conversation_id: UUID,
        turn_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}"
                f"/turns/{turn_id}"
            ),
            auth_context=auth_context,
        )

    async def list_conversation_turn_events(
        self,
        conversation_id: UUID,
        turn_id: UUID,
        *,
        after_sequence: int,
        limit: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}"
                f"/turns/{turn_id}/events?after_sequence={after_sequence}"
                f"&limit={limit}"
            ),
            auth_context=auth_context,
        )

    async def cancel_conversation_turn(
        self,
        conversation_id: UUID,
        turn_id: UUID,
        *,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/aiops/conversations/{conversation_id}"
                f"/turns/{turn_id}/cancel"
            ),
            payload={},
            auth_context=auth_context,
        )

    async def report_template_request(
        self,
        method: str,
        suffix: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
    ):
        return await self._json(
            method,
            f"{INTERNAL_API_V1}/aiops/report-templates{suffix}",
            payload=payload,
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
