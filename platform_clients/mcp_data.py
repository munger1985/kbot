"""SelectAI/AIReport 问数服务的受限 HTTP Client。"""

from __future__ import annotations

from typing import Any

import aiohttp


class MCPDataClientError(RuntimeError):
    """外部问数服务返回不可用或不合法结果。"""


class MCPDataClient:
    def __init__(
        self,
        *,
        api_endpoint: str,
        profiles_endpoint: str,
        api_key: str,
        timeout_seconds: int,
        max_rows: int,
        max_response_bytes: int,
        session: aiohttp.ClientSession | None = None,
    ):
        self._api_endpoint = api_endpoint
        self._profiles_endpoint = profiles_endpoint
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_rows = max_rows
        self._max_response_bytes = max_response_bytes
        self._session = session

    async def query(
        self, *, profile: str, user: str, question: str
    ) -> dict[str, Any]:
        payload = {"profile": profile, "user": user, "ask": question}
        response = await self._request(
            "POST", self._api_endpoint, json=payload
        )
        if not isinstance(response, dict):
            raise MCPDataClientError("问数服务响应不是 JSON 对象")
        data = response.get("data")
        if not isinstance(data, list):
            raise MCPDataClientError("问数服务响应缺少 data 数组")
        first = data[0] if data else {}
        rows = first.get("data", []) if isinstance(first, dict) else []
        if not isinstance(rows, list):
            raise MCPDataClientError("问数服务 data[0].data 不是数组")
        truncated = len(rows) > self._max_rows
        return {
            "rows": rows[: self._max_rows],
            "row_count": min(len(rows), self._max_rows),
            "upstream_row_count": len(rows),
            "truncated": truncated,
        }

    async def list_profiles(self) -> Any:
        return await self._request("GET", self._profiles_endpoint)

    async def _request(
        self, method: str, url: str, *, json: dict | None = None
    ) -> Any:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(
            timeout=self._timeout
        )
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        try:
            async with session.request(
                method,
                url,
                json=json,
                headers=headers,
                timeout=self._timeout,
            ) as response:
                if response.status != 200:
                    detail = (await response.text())[:1000]
                    raise MCPDataClientError(
                        f"问数服务返回 HTTP {response.status}：{detail}"
                    )
                content_length = response.content_length
                if (
                    content_length is not None
                    and content_length > self._max_response_bytes
                ):
                    raise MCPDataClientError("问数服务响应超过大小上限")
                raw = await response.read()
                if len(raw) > self._max_response_bytes:
                    raise MCPDataClientError("问数服务响应超过大小上限")
                try:
                    return await response.json(
                        encoding=response.charset or "utf-8"
                    )
                except (ValueError, UnicodeDecodeError) as exc:
                    raise MCPDataClientError(
                        "问数服务返回的不是合法 JSON"
                    ) from exc
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise MCPDataClientError("无法连接外部问数服务") from exc
        finally:
            if owns_session:
                await session.close()
