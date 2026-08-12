"""Asset MetaDB HTTP Client。"""

import base64
from datetime import datetime
from typing import Any

import aiohttp


class AssetMetaDbError(RuntimeError):
    pass


class AssetMetaDbClient:
    def __init__(self, *, endpoint: str, username: str, password: str, timeout_seconds: int = 30):
        self._endpoint = endpoint.rstrip("?")
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self._headers = {"Authorization": f"Basic {token}", "Accept": "application/json"}
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def list_assets(self, *, offset: int, limit: int, processed: str = "N") -> list[dict[str, Any]]:
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(self._endpoint, params={"offset": offset, "limit": limit, "processed": processed.upper()}, headers=self._headers) as response:
                    body = await response.json(content_type=None)
                    if response.status >= 400:
                        raise AssetMetaDbError(f"MetaDB 查询失败：HTTP {response.status}")
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise AssetMetaDbError("MetaDB 暂时不可用") from exc
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            raise AssetMetaDbError("MetaDB 响应缺少 items 数组")
        return [dict(item) for item in body["items"] if isinstance(item, dict)]

    async def set_processed(self, *, asset_id: str, processed: str) -> None:
        payload = {
            "asset_id": asset_id,
            "processed": processed.upper(),
            "sp_file_name": "",
            "sp_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.put(self._endpoint, json=payload, headers={**self._headers, "Content-Type": "application/json"}) as response:
                    if response.status >= 400:
                        raise AssetMetaDbError(f"MetaDB 状态更新失败：HTTP {response.status}")
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AssetMetaDbError("MetaDB 状态更新失败") from exc
