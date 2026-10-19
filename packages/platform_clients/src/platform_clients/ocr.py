"""Model Serving OCR 内部客户端。"""

import base64
from typing import Any
from uuid import UUID

import aiohttp

from platform_core.contracts import AuthContext, INTERNAL_API_V1
from platform_core.security import build_internal_auth_headers


class OCRClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OCRClient:
    def __init__(
        self, *, base_url: str, caller_service: str, audience: str,
        timeout_seconds: int = 120,
    ):
        self._base_url = base_url.rstrip("/")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def infer(
        self, *, model_id: UUID, image: bytes, mime_type: str,
        auth_context: AuthContext,
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            **build_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
                context=auth_context,
            ),
        }
        payload = {
            "model_id": str(model_id),
            "image_base64": base64.b64encode(image).decode("ascii"),
            "mime_type": mime_type,
        }
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                f"{self._base_url}{INTERNAL_API_V1}/inference",
                json=payload,
                headers=headers,
            ) as response:
                result = await response.json(content_type=None)
                if response.status >= 400:
                    detail = result.get("detail", result) if isinstance(result, dict) else {}
                    raise OCRClientError(
                        str(detail.get("code", "OCR_UNAVAILABLE")),
                        str(detail),
                    )
                return result
