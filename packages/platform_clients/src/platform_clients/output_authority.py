"""AIOps 后台输出的窄 Main API 授权客户端；每次调用签发新的双 JWT。"""

import aiohttp

from platform_core.contracts.aiops.output_authority import OutputAuthorityRequest, OutputAuthoritySnapshot
from platform_core.security import build_scoped_internal_auth_headers, create_service_auth_context


class OutputAuthorityClientError(RuntimeError):
    def __init__(self, *, status_code: int, retryable: bool):
        super().__init__("当前后台输出授权暂不可用" if retryable else "当前后台输出授权已失效")
        self.status_code, self.retryable = status_code, retryable


class OutputAuthorityClient:
    def __init__(self, *, base_url: str, audience: str, caller_service: str,
        session: aiohttp.ClientSession, timeout_seconds: float = 10):
        self._base_url = base_url.rstrip("/")
        self._audience, self._caller = audience, caller_service
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def resolve(self, request: OutputAuthorityRequest, *, trace_id: str) -> OutputAuthoritySnapshot:
        context = create_service_auth_context(caller_service=self._caller, trace_id=trace_id)
        headers = build_scoped_internal_auth_headers(
            audience=self._audience, caller_service=self._caller,
            scopes=("aiops.output.authorize",), context=context,
        )
        try:
            async with self._session.post(
                f"{self._base_url}/internal/v1/platform/aiops/output-authority",
                headers=headers, json=request.model_dump(mode="json"), timeout=self._timeout,
            ) as response:
                if response.status != 200:
                    # 不读取或转储上游错误正文，令牌和输入也不进入日志。
                    raise OutputAuthorityClientError(status_code=response.status,
                        retryable=response.status >= 500 or response.status == 429)
                snapshot = OutputAuthoritySnapshot.model_validate(await response.json())
                if snapshot.request != request:
                    raise OutputAuthorityClientError(status_code=502, retryable=False)
                return snapshot
        except OutputAuthorityClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise OutputAuthorityClientError(status_code=503, retryable=True) from exc
