"""模型服务 Responses 内部 Client：X Search 与文生图。"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from platform_core.contracts import (
    GENERATIVE_ERROR_CODES,
    INTERNAL_API_V1,
    ImageGenerationRequest,
    ImageGenerationResult,
    ResearchRequest,
    ResearchResult,
)
from platform_core.security import build_internal_auth_headers


class GenerativeResponsesClientError(RuntimeError):
    """将模型服务生成失败映射为稳定错误码。"""

    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class GenerativeResponsesClient:
    """只暴露研究与文生图调用，不把能力验收入口透传到 Main API。"""

    def __init__(
        self,
        *,
        base_url: str,
        caller_service: str,
        audience: str,
        timeout_seconds: int = 300,
        session: aiohttp.ClientSession | None = None,
    ):
        resolved = base_url.rstrip("/")
        self._base_url = resolved.replace("://0.0.0.0", "://127.0.0.1")
        self._caller_service = caller_service
        self._audience = audience
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session = session

    async def research(self, request: ResearchRequest) -> ResearchResult:
        payload = await self._json(
            "POST",
            f"{INTERNAL_API_V1}/responses/research",
            body=request.model_dump(mode="json"),
        )
        return ResearchResult.model_validate(payload)

    async def generate_image(
        self, request: ImageGenerationRequest,
    ) -> ImageGenerationResult:
        payload = await self._json(
            "POST",
            f"{INTERNAL_API_V1}/responses/image-generations",
            body=request.model_dump(mode="json"),
        )
        return ImageGenerationResult.model_validate(payload)

    async def _json(
        self, method: str, path: str, *, body: dict[str, Any] | None = None,
    ) -> Any:
        owns_session = self._session is None
        session = self._session or aiohttp.ClientSession(timeout=self._timeout)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **build_internal_auth_headers(
                audience=self._audience,
                caller_service=self._caller_service,
            ),
        }
        try:
            async with session.request(
                method, f"{self._base_url}{path}", headers=headers, json=body,
            ) as response:
                raw = await response.text()
                payload: Any = {}
                if raw:
                    try:
                        payload = await response.json(content_type=None)
                    except Exception:
                        payload = {"message": raw}
                if response.status >= 400:
                    raise self._error(response.status, payload)
                return payload
        except GenerativeResponsesClientError:
            raise
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ServerTimeoutError) as exc:
            raise GenerativeResponsesClientError(
                status_code=504,
                code="PROVIDER_TIMEOUT",
                message="模型服务生成请求超时",
            ) from exc
        except (aiohttp.ClientError, OSError) as exc:
            raise GenerativeResponsesClientError(
                status_code=503,
                code="PROVIDER_UNAVAILABLE",
                message="模型服务暂时不可用",
            ) from exc
        finally:
            if owns_session:
                await session.close()

    @staticmethod
    def _error(status_code: int, payload: Any) -> GenerativeResponsesClientError:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        if not isinstance(detail, dict):
            detail = {"message": str(detail)}
        code = str(detail.get("code") or "")
        if code not in GENERATIVE_ERROR_CODES:
            if status_code in {408, 504}:
                code = "PROVIDER_TIMEOUT"
            else:
                code = "PROVIDER_UNAVAILABLE"
        message = str(
            detail.get("message")
            or detail.get("detail")
            or "模型服务生成请求失败"
        )
        return GenerativeResponsesClientError(
            status_code=status_code, code=code, message=message,
        )
