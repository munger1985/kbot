"""App API 的有界 JSON 读取；不持有数据库会话。"""

import json
from fastapi import HTTPException


async def bounded_request(request, *, max_bytes=65536):
    value = bytearray()
    async for block in request.stream():
        if len(value) + len(block) > max_bytes:
            raise HTTPException(413, detail={"code": "RESOURCE_LIMIT"})
        value.extend(block)
    try:
        result = json.loads(value)
        if not isinstance(result, dict):
            raise ValueError()
        return result
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, detail={"code": "INVALID_CONTRACT"}) from exc
