"""KM Portal 公开 Main API Client。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import PUBLIC_API_V1


class KmPortalClientError(RuntimeError):
    """KM Portal 公开接口调用失败。"""

    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class KmPortalClient:
    """复用 KM Portal 浏览器使用的公开聊天接口。"""

    _BASE = f"{PUBLIC_API_V1}/apps/km-asset"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 120,
        session: aiohttp.ClientSession | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def create_conversation(self, *, payload: dict[str, Any]):
        return await self._json("POST", "/conversations", payload=payload)

    async def get_conversation(self, *, conversation_id: UUID):
        return await self._json("GET", f"/conversations/{conversation_id}")

    async def create_conversation_turn(
        self,
        *,
        conversation_id: UUID,
        payload: dict[str, Any],
        idempotency_key: str,
    ):
        return await self._json(
            "POST",
            f"/conversations/{conversation_id}/turns",
            payload=payload,
            extra_headers={"Idempotency-Key": idempotency_key},
        )

    async def get_run(self, *, run_id: UUID):
        return await self._json("GET", f"/runs/{run_id}")

    async def get_result(self, *, run_id: UUID):
        return await self._json("GET", f"/runs/{run_id}/result")

    async def get_reference_preview(
        self, *, run_id: UUID, citation_label: str
    ):
        return await self._json(
            "GET",
            f"/runs/{run_id}/references/{citation_label}/preview",
        )

    async def _json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **(extra_headers or {}),
        }
        try:
            async with session.request(
                method,
                f"{self._base_url}{self._BASE}{path}",
                headers=headers,
                json=payload,
            ) as response:
                try:
                    body = await response.json()
                except (aiohttp.ContentTypeError, ValueError):
                    body = {"detail": await response.text()}
                if response.status >= 400:
                    detail = (
                        body.get("detail", body)
                        if isinstance(body, dict)
                        else body
                    )
                    code = (
                        str(detail.get("code") or "KM_PORTAL_API_ERROR")
                        if isinstance(detail, dict)
                        else "KM_PORTAL_API_ERROR"
                    )
                    message = (
                        str(detail.get("message") or detail)
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    raise KmPortalClientError(
                        status_code=response.status,
                        code=code,
                        message=message,
                    )
                return body
        except KmPortalClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise KmPortalClientError(
                status_code=503,
                code="KM_PORTAL_API_UNAVAILABLE",
                message="KM Portal Main API 暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()


__all__ = ["KmPortalClient", "KmPortalClientError"]
