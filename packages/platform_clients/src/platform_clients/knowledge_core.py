"""Knowledge Core 的版本化内部 HTTP Client。"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class KnowledgeCoreClientError(RuntimeError):
    """KC 返回的稳定上游错误。"""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True, slots=True)
class KnowledgeCoreResponse:
    status_code: int
    payload: Any


@dataclass(frozen=True, slots=True)
class KnowledgeCoreStreamResponse:
    """KC 二进制响应；body 消费结束后自动释放 HTTP 连接。"""

    status_code: int
    headers: dict[str, str]
    body: AsyncIterable[bytes]


def _json_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {
            str(_json_value(key)): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class KnowledgeCoreClient:
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

    async def list_collections(
        self,
        *,
        domain_id: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}/collections",
            auth_context=auth_context,
        )

    async def is_ready(self) -> bool:
        """查询 KC 就绪端点，不携带内部或门户凭据。"""
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        try:
            async with session.get(f"{self._base_url}/readyz") as response:
                return response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False
        finally:
            if owns_session:
                await session.close()

    async def create_collection(
        self,
        *,
        domain_id: int,
        payload: dict[str, Any],
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}/collections",
            payload=payload,
            auth_context=auth_context,
        )

    async def get_collection(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}"
            ),
            auth_context=auth_context,
        )

    async def change_collection_status(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        status: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PATCH",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}"
            ),
            payload={"status": status},
            auth_context=auth_context,
        )

    async def update_collection_models(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        payload: dict[str, Any],
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PUT",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/models"
            ),
            payload=payload,
            auth_context=auth_context,
        )

    async def list_model_references(
        self,
        *,
        model_id: UUID,
        auth_context: AuthContext,
    ) -> list[dict[str, Any]]:
        payload = await self._json(
            "GET",
            f"{INTERNAL_API_V1}/knowledge/model-references/{model_id}",
            auth_context=auth_context,
        )
        references = payload.get("references", [])
        if not isinstance(references, list):
            raise KnowledgeCoreClientError(
                status_code=502,
                code="KNOWLEDGE_CORE_PROTOCOL_ERROR",
                message="Knowledge Core 模型引用响应无效",
            )
        return references

    async def delete_collection(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "DELETE",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}"
            ),
            auth_context=auth_context,
        )

    async def bind_collection(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        collection_id: UUID,
        note: str | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "PUT",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/agents/{agent_id}/collections/{collection_id}/binding"
            ),
            payload={"note": note},
            auth_context=auth_context,
        )

    async def unbind_collection(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        collection_id: UUID,
        auth_context: AuthContext,
    ) -> None:
        await self._json(
            "DELETE",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/agents/{agent_id}/collections/{collection_id}/binding"
            ),
            auth_context=auth_context,
            allow_empty=True,
        )

    async def list_agent_bindings(
        self,
        *,
        domain_id: int,
        agent_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/agents/{agent_id}/collection-bindings"
            ),
            auth_context=auth_context,
        )

    async def list_collection_bindings(
        self, *, domain_id: int, collection_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/bindings"
            ),
            auth_context=auth_context,
        )

    async def get_bundle_status(
        self,
        *,
        domain_id: int,
        bundle_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/bundles/{bundle_id}"
            ),
            auth_context=auth_context,
        )

    async def get_revision_status(
        self,
        *,
        domain_id: int,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        include_members: bool,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        suffix = "/members" if include_members else ""
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/bundles/{bundle_id}/revisions/{bundle_revision_id}{suffix}"
            ),
            auth_context=auth_context,
        )

    async def list_processing(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        page: int,
        page_size: int,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/catalog/processing?collection_id={collection_id}"
                f"&page={page}&page_size={page_size}"
            ),
            auth_context=auth_context,
        )

    async def get_bundle_revision_preview(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/bundles/{bundle_id}"
                f"/revisions/{bundle_revision_id}/preview"
            ),
            auth_context=auth_context,
        )

    async def stream_source_file(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        document_version_id: UUID,
        range_header: str | None,
        auth_context: AuthContext,
    ) -> KnowledgeCoreStreamResponse:
        headers = self._headers(auth_context)
        headers["Accept"] = "*/*"
        if range_header:
            headers["Range"] = range_header
        return await self._stream(
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/bundles/{bundle_id}"
                f"/revisions/{bundle_revision_id}/documents/"
                f"{document_version_id}/content"
            ),
            headers=headers,
        )

    async def reprocess_revision(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        document_version_id: UUID | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/bundles/{bundle_id}/revisions/{bundle_revision_id}"
                "/reprocess"
            ),
            payload={
                "collection_id": collection_id,
                "document_version_id": document_version_id,
            },
            auth_context=auth_context,
        )

    async def reindex_discovery(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        """调度 KC Discovery 全文与向量索引重建。"""
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/bundles/{bundle_id}/revisions/{bundle_revision_id}"
                "/reindex-discovery"
            ),
            payload={"collection_id": collection_id},
            auth_context=auth_context,
        )

    async def get_reindex_discovery_status(
        self,
        *,
        domain_id: int,
        bundle_id: UUID,
        bundle_revision_id: UUID,
        generation: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        """查询 KC Discovery 重建操作状态。"""
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/bundles/{bundle_id}/revisions/{bundle_revision_id}"
                f"/reindex-discovery/{generation}"
            ),
            auth_context=auth_context,
        )

    async def list_pending_approvals(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "GET",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/approvals"
            ),
            auth_context=auth_context,
        )

    async def review_user_intake(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        bundle_revision_id: UUID,
        decision: str,
        comment: str | None,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            (
                f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
                f"/collections/{collection_id}/bundle-revisions/"
                f"{bundle_revision_id}/approval"
            ),
            payload={"decision": decision, "comment": comment},
            auth_context=auth_context,
        )

    async def ingest_multipart(
        self,
        *,
        domain_id: int,
        collection_id: UUID,
        intake_kind: str,
        content_type: str,
        body: AsyncIterable[bytes],
        idempotency_key: str,
        auth_context: AuthContext,
    ) -> KnowledgeCoreResponse:
        if intake_kind not in {"km-assets", "user-files"}:
            raise ValueError("不支持的 KC 入库类型")
        path = (
            f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}"
            f"/collections/{collection_id}/ingestions/{intake_kind}"
        )
        headers = self._headers(auth_context)
        headers["Content-Type"] = content_type
        headers["Idempotency-Key"] = idempotency_key
        return await self._raw(
            "POST",
            path,
            headers=headers,
            data=body,
        )

    async def discover(
        self,
        *,
        query: str,
        collection_ids: Sequence[UUID],
        domain_id: int,
        agent_id: str,
        auth_context: AuthContext,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        max_security_level: int = 3,
        per_collection_limit: int = 20,
        coverage_mode: str = "BALANCED",
        run_id: UUID | None = None,
        task_id: UUID | None = None,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/knowledge/discovery/search",
            payload={
                "query": query,
                "collection_ids": [str(value) for value in collection_ids],
                "domain_id": domain_id,
                "agent_id": agent_id,
                "query_vectors": (
                    {
                        str(key): list(value)
                        for key, value in query_vectors.items()
                    }
                    if query_vectors
                    else None
                ),
                "max_security_level": max_security_level,
                "per_collection_limit": per_collection_limit,
                "coverage_mode": coverage_mode,
                "run_id": str(run_id) if run_id else None,
                "task_id": str(task_id) if task_id else None,
            },
            auth_context=auth_context,
        )

    async def retrieve_evidence(
        self,
        *,
        query: str,
        candidates: Sequence[dict[str, Any]],
        domain_id: int,
        agent_id: str,
        auth_context: AuthContext,
        query_vectors: dict[UUID, Sequence[float]] | None = None,
        max_security_level: int = 3,
        max_evidence: int = 12,
        context_limit: int = 4,
        coverage_mode: str = "BALANCED",
        run_id: UUID | None = None,
        task_id: UUID | None = None,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/knowledge/retrieval/evidence",
            payload={
                "query": query,
                "candidates": list(candidates),
                "domain_id": domain_id,
                "agent_id": agent_id,
                "query_vectors": (
                    {
                        str(key): list(value)
                        for key, value in query_vectors.items()
                    }
                    if query_vectors
                    else None
                ),
                "max_security_level": max_security_level,
                "max_evidence": max_evidence,
                "context_limit": context_limit,
                "coverage_mode": coverage_mode,
                "run_id": str(run_id) if run_id else None,
                "task_id": str(task_id) if task_id else None,
            },
            auth_context=auth_context,
        )

    async def search_visual(
        self,
        *,
        images_base64: Sequence[str],
        collection_ids: Sequence[UUID],
        domain_id: int,
        agent_id: UUID,
        auth_context: AuthContext,
        per_image_limit: int = 10,
        result_limit: int = 20,
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"{INTERNAL_API_V1}/knowledge/retrieval/visual",
            payload={
                "domain_id": domain_id,
                "agent_id": str(agent_id),
                "collection_ids": [
                    str(value) for value in collection_ids
                ],
                "images_base64": list(images_base64),
                "per_image_limit": per_image_limit,
                "result_limit": result_limit,
            },
            auth_context=auth_context,
        )

    async def _json(
        self,
        method: str,
        path: str,
        *,
        auth_context: AuthContext,
        payload: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> Any:
        response = await self._raw(
            method,
            path,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                **self._headers(auth_context),
            },
            json=_json_value(payload),
        )
        if response.payload is None and allow_empty:
            return None
        return response.payload

    async def _raw(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        **kwargs,
    ) -> KnowledgeCoreResponse:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        try:
            async with session.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                **kwargs,
            ) as response:
                payload = await self._response_payload(response)
                if response.status >= 400:
                    detail = (
                        payload.get("detail", payload)
                        if isinstance(payload, dict)
                        else payload
                    )
                    code = (
                        detail.get("code", "KNOWLEDGE_CORE_ERROR")
                        if isinstance(detail, dict)
                        else "KNOWLEDGE_CORE_ERROR"
                    )
                    message = (
                        detail.get("message", str(detail))
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    raise KnowledgeCoreClientError(
                        status_code=response.status,
                        code=code,
                        message=message,
                    )
                return KnowledgeCoreResponse(
                    status_code=response.status,
                    payload=payload,
                )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise KnowledgeCoreClientError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()

    async def _stream(
        self,
        path: str,
        *,
        headers: dict[str, str],
    ) -> KnowledgeCoreStreamResponse:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        try:
            response = await session.get(
                f"{self._base_url}{path}", headers=headers
            )
        except (aiohttp.ClientError, TimeoutError) as exc:
            if owns_session:
                await session.close()
            raise KnowledgeCoreClientError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 暂时不可用",
            ) from exc
        if response.status >= 400:
            try:
                payload = await self._response_payload(response)
                detail = (
                    payload.get("detail", payload)
                    if isinstance(payload, dict)
                    else payload
                )
                code = (
                    detail.get("code", "KNOWLEDGE_CORE_ERROR")
                    if isinstance(detail, dict)
                    else "KNOWLEDGE_CORE_ERROR"
                )
                message = (
                    detail.get("message", str(detail))
                    if isinstance(detail, dict)
                    else str(detail)
                )
            finally:
                response.release()
                if owns_session:
                    await session.close()
            raise KnowledgeCoreClientError(
                status_code=response.status,
                code=code,
                message=message,
            )

        async def iter_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.content.iter_chunked(64 * 1024):
                    yield chunk
            finally:
                response.release()
                if owns_session:
                    await session.close()

        return KnowledgeCoreStreamResponse(
            status_code=response.status,
            headers={
                key.lower(): value
                for key, value in response.headers.items()
            },
            body=iter_body(),
        )

    @staticmethod
    async def _response_payload(response: aiohttp.ClientResponse) -> Any:
        if response.status == 204:
            return None
        try:
            return await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            return {"message": await response.text()}

    def _headers(self, auth_context: AuthContext) -> dict[str, str]:
        return build_internal_auth_headers(
            audience=self._audience,
            caller_service=self._caller_service,
            context=auth_context,
        )
