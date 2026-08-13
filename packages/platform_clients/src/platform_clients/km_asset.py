"""KM Asset App 内部 Client。"""

import json
from typing import Any
from urllib.parse import quote, urlencode
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext
from platform_core.security import build_scoped_internal_auth_headers


class KmAssetClientError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class KmAssetClient:
    _BASE = "/internal/v1/km-asset"

    def __init__(self, *, base_url: str, caller_service: str, audience: str, timeout_seconds: int = 120, session: aiohttp.ClientSession | None = None):
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def list_sources(self, *, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/sources?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def create_source(self, *, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/sources", payload=payload, auth_context=auth_context)

    async def update_source(self, *, source_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("PATCH", f"{self._BASE}/sources/{source_id}", payload=payload, auth_context=auth_context)

    async def activate_source(self, *, source_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/sources/{source_id}/activate", payload=payload, auth_context=auth_context)

    async def sync_source(self, *, source_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/sources/{source_id}/sync?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def list_metadb_assets(self, *, source_id: UUID, domain_id: int, processed: str, offset: int, limit: int, auth_context: AuthContext):
        query = urlencode({"domain_id": domain_id, "processed": processed, "offset": offset, "limit": limit})
        return await self._json("GET", f"{self._BASE}/sources/{source_id}/metadb/assets?{query}", auth_context=auth_context)

    async def retry_metadb_asset(self, *, source_id: UUID, external_asset_id: str, domain_id: int, auth_context: AuthContext):
        query = urlencode({"domain_id": domain_id})
        encoded_asset_id = quote(external_asset_id, safe="")
        return await self._json("POST", f"{self._BASE}/sources/{source_id}/metadb/assets/{encoded_asset_id}/retry?{query}", auth_context=auth_context)

    async def data_model(self, *, source_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/sources/{source_id}/data-model?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def reconcile_data_model(self, *, source_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/sources/{source_id}/data-model/reconcile?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def list_assets(self, *, domain_id: int, source_id: UUID | None, ingestion_status: str | None, offset: int, limit: int, auth_context: AuthContext):
        values: dict[str, Any] = {"domain_id": domain_id, "offset": offset, "limit": limit}
        if source_id is not None:
            values["source_id"] = source_id
        if ingestion_status is not None:
            values["ingestion_status"] = ingestion_status
        return await self._json("GET", f"{self._BASE}/assets?{urlencode(values)}", auth_context=auth_context)

    async def get_asset(self, *, km_asset_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/assets/{km_asset_id}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def retry_asset(self, *, km_asset_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/assets/{km_asset_id}/retry", payload=payload, auth_context=auth_context)

    async def reindex_asset(self, *, km_asset_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/assets/{km_asset_id}/reindex", payload=payload, auth_context=auth_context)

    async def list_jobs(self, *, domain_id: int, source_id: UUID | None, limit: int, auth_context: AuthContext):
        values: dict[str, Any] = {"domain_id": domain_id, "limit": limit}
        if source_id is not None:
            values["source_id"] = source_id
        return await self._json("GET", f"{self._BASE}/jobs?{urlencode(values)}", auth_context=auth_context)

    async def list_agents(self, *, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/agents?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def create_agent(self, *, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/agents", payload=payload, auth_context=auth_context)

    async def get_agent(self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/agents/{agent_id}?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def update_agent(self, *, agent_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("PATCH", f"{self._BASE}/agents/{agent_id}", payload=payload, auth_context=auth_context)

    async def activate_agent(self, *, agent_id: UUID, payload: dict[str, Any], auth_context: AuthContext):
        return await self._json("POST", f"{self._BASE}/agents/{agent_id}/activate", payload=payload, auth_context=auth_context)

    async def execution_spec(self, *, agent_id: UUID, domain_id: int, auth_context: AuthContext):
        return await self._json("GET", f"{self._BASE}/agents/{agent_id}/execution-spec?{urlencode({'domain_id': domain_id})}", auth_context=auth_context)

    async def _json(self, method: str, path: str, *, auth_context: AuthContext, payload: dict[str, Any] | None = None):
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {"Accept": "application/json", **build_scoped_internal_auth_headers(audience=self._audience, caller_service=self._caller_service, scopes=("km_asset.manage",), context=auth_context)}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        try:
            async with session.request(method, f"{self._base_url}{path}", headers=headers, json=payload) as response:
                response_text = await response.text()
                try:
                    body = json.loads(response_text) if response_text else None
                except json.JSONDecodeError:
                    body = response_text
                if response.status >= 400:
                    detail = body.get("detail", body) if isinstance(body, dict) else body
                    code = str(detail.get("code", "KM_ASSET_APP_ERROR")) if isinstance(detail, dict) else "KM_ASSET_APP_ERROR"
                    message = str(detail.get("message", detail)) if isinstance(detail, dict) else str(detail or f"HTTP {response.status}")
                    raise KmAssetClientError(status_code=response.status, code=code, message=message)
                return body
        except KmAssetClientError:
            raise
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise KmAssetClientError(status_code=503, code="KM_ASSET_APP_UNAVAILABLE", message="KM Asset App 暂时不可用") from exc
        finally:
            if owns_session:
                await session.close()
