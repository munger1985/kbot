"""Data Query Service 的窄内部客户端。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext
from platform_core.security import build_scoped_internal_auth_headers


class DataQueryClientError(RuntimeError):
    """Data Query 服务不可用或返回不合法响应。"""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class DataQueryClient:
    """Data Query 的窄内部 Client，不提供任意 SQL 或 Secret 操作。"""

    def __init__(
        self,
        *,
        base_url: str,
        caller_service: str = "kbot-agent-runtime-api",
        audience: str = "kbot-data-query-api",
        timeout_seconds: int = 120,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
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

    async def create_run(
        self, *, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        """由 Agent Runtime 委派受控 Query Plan；服务端仍会重新验证 Binding。"""
        return await self._json(
            "POST", "/internal/v1/data-query/runs", payload=payload,
            auth_context=auth_context, scopes=("data_query.delegate",),
        )

    async def get_planning_context(
        self, *, consumer_app_id: str, agent_id: UUID,
        agent_version_id: UUID, auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET", (
                f"/internal/v1/data-query/runs/planning-context/{agent_id}"
                f"?consumer_app_id={consumer_app_id}"
                f"&agent_version_id={agent_version_id}"
            ),
            auth_context=auth_context, scopes=("data_query.delegate",),
        )

    async def get_run(self, *, data_query_run_id: UUID, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json(
            "GET", f"/internal/v1/data-query/runs/{data_query_run_id}",
            auth_context=auth_context, scopes=("data_query.delegate",),
        )

    async def get_result(self, *, data_query_run_id: UUID, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json(
            "GET", f"/internal/v1/data-query/runs/{data_query_run_id}/result",
            auth_context=auth_context, scopes=("data_query.delegate",),
        )

    async def cancel_run(self, *, data_query_run_id: UUID, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json(
            "POST", f"/internal/v1/data-query/runs/{data_query_run_id}/cancel",
            auth_context=auth_context, scopes=("data_query.delegate",),
        )

    async def management_list(
        self, *, resource: str, cursor: UUID | None, limit: int, auth_context: AuthContext
    ) -> dict[str, Any]:
        suffix = f"?limit={limit}" + (f"&cursor={cursor}" if cursor else "")
        return await self._json(
            "GET", f"/internal/v1/data-query/management/{resource}{suffix}",
            auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_capabilities(self, *, auth_context: AuthContext) -> dict[str, Any]:
        return await self._json(
            "GET", "/internal/v1/data-query/management/connector-capabilities",
            auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def reconcile_km_asset_dataset(
        self, *, auth_context: AuthContext
    ) -> dict[str, Any]:
        """由 KM Asset App 调和系统托管问数资源。"""
        return await self._json(
            "POST",
            "/internal/v1/data-query/managed-datasets/km-asset/reconcile",
            payload={},
            auth_context=auth_context,
            scopes=("data_query.managed",),
        )

    async def management_get(
        self, *, resource: str, resource_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "GET", f"/internal/v1/data-query/management/{resource}/{resource_id}",
            auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_update(
        self, *, resource: str, resource_id: UUID, payload: dict[str, Any],
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PUT", f"/internal/v1/data-query/management/{resource}/{resource_id}",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_change_status(
        self, *, resource: str, resource_id: UUID, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH", f"/internal/v1/data-query/management/{resource}/{resource_id}/status",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_create(
        self, *, resource: str, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST", f"/internal/v1/data-query/management/{resource}", payload=payload,
            auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_has_active_agent_binding(
        self, *, consumer_app_id: str, agent_id: UUID,
        agent_version_id: UUID, semantic_model_ids: set[UUID],
        auth_context: AuthContext,
    ) -> bool:
        result = await self._json(
            "POST",
            "/internal/v1/data-query/management/agent-bindings/active-match",
            payload={
                "consumer_app_id": consumer_app_id,
                "agent_id": str(agent_id),
                "agent_version_id": str(agent_version_id),
                "semantic_model_ids": [str(item) for item in sorted(semantic_model_ids)],
            },
            auth_context=auth_context,
            scopes=("data_query.manage",),
        )
        return result.get("matched") is True

    async def management_test_connection(
        self, *, payload: dict[str, Any], auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/internal/v1/data-query/management/data-sources/test-connection",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_request_snapshot(
        self, *, data_source_id: UUID, auth_context: AuthContext
    ) -> dict[str, Any]:
        return await self._json(
            "POST", f"/internal/v1/data-query/management/data-sources/{data_source_id}/snapshots",
            payload={}, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_action(
        self, *, method: str, path: str, payload: dict[str, Any] | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        """调用固定管理动作；path 只能由 BFF 代码构造，不能接受浏览器原样输入。"""
        return await self._json(
            method, f"/internal/v1/data-query/management/{path.lstrip('/')}",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def list_model_references(
        self, *, model_id: UUID, auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        payload = await self._json(
            "GET", f"/internal/v1/data-query/model-references/{model_id}",
            auth_context=auth_context, scopes=("model.references",),
        )
        references = payload.get("references", [])
        if not isinstance(references, list):
            raise DataQueryClientError(
                status_code=502,
                code="DATA_QUERY_REFERENCE_RESPONSE_INVALID",
                message="Data Query 返回了无效模型引用",
            )
        return [item for item in references if isinstance(item, dict)]

    async def management_publish_model(
        self, *, semantic_model_id: UUID, semantic_model_version_id: UUID,
        payload: dict[str, Any], auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/internal/v1/data-query/management/semantic-models/"
            f"{semantic_model_id}/versions/{semantic_model_version_id}/publish",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_submit_model_review(
        self, *, semantic_model_id: UUID, semantic_model_version_id: UUID,
        payload: dict[str, Any], auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/internal/v1/data-query/management/semantic-models/"
            f"{semantic_model_id}/versions/{semantic_model_version_id}/submit-review",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def management_return_model_for_revision(
        self, *, semantic_model_id: UUID, semantic_model_version_id: UUID,
        payload: dict[str, Any], auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/internal/v1/data-query/management/semantic-models/"
            f"{semantic_model_id}/versions/{semantic_model_version_id}/return-for-revision",
            payload=payload, auth_context=auth_context, scopes=("data_query.manage",),
        )

    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        scopes: tuple[str, ...],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = build_scoped_internal_auth_headers(
            audience=self._audience,
            context=auth_context,
            caller_service=self._caller_service,
            scopes=scopes,
        )
        try:
            async with session.request(
                method, f"{self._base_url}{path}", json=payload, headers=headers
            ) as response:
                if response.status == 204:
                    return {}
                body = await response.json(content_type=None)
                if response.status >= 400:
                    detail = body if isinstance(body, dict) else {}
                    raise DataQueryClientError(
                        status_code=response.status,
                        code=str(detail.get("code", "DATA_QUERY_REQUEST_FAILED")),
                        message=str(detail.get("detail", detail.get("code", "Data Query 请求失败"))),
                    )
                if not isinstance(body, dict):
                    raise DataQueryClientError(status_code=502, code="DATA_QUERY_INVALID_RESPONSE", message="Data Query 返回不是 JSON Object")
                return body
        except aiohttp.ClientError as exc:
            raise DataQueryClientError(status_code=503, code="DATA_QUERY_UNAVAILABLE", message="Data Query 服务不可用") from exc

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            self._owns_session = True
        return self._session
