"""系统案例资源客户端；不转发用户凭证，每次请求签发短期双 JWT。"""

import aiohttp

from platform_core.contracts.aiops.case_resource import (
    CASE_RESOURCE_PATH, CASE_RESOURCE_SCOPE, CaseResourceRequest, CaseResourceResult,
)
from platform_core.security import build_scoped_internal_auth_headers, create_service_auth_context


class CaseResourceClientError(RuntimeError):
    def __init__(self, code="CASE_RESOURCE_UNAVAILABLE", *, retryable=True):
        super().__init__(code)
        self.code, self.retryable = code, retryable


class CaseResourceClient:
    def __init__(self, *, base_url, audience, caller_service, session):
        self._url = base_url.rstrip("/") + CASE_RESOURCE_PATH
        self._audience, self._caller, self._session = audience, caller_service, session

    async def ensure(self, request: CaseResourceRequest) -> CaseResourceResult:
        context = create_service_auth_context(caller_service=self._caller)
        headers = build_scoped_internal_auth_headers(audience=self._audience,
            caller_service=self._caller, scopes=(CASE_RESOURCE_SCOPE,), context=context)
        try:
            async with self._session.post(self._url, headers=headers,
                json=request.model_dump(mode="json"), timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    code = "CASE_RESOURCE_UNAVAILABLE"
                    if response.status in {409, 422}:
                        body = await response.json()
                        code = (body.get("detail") or {}).get("code", "CASE_RESOURCE_CONFLICT")
                        if code not in {"CASE_RESOURCE_CONFLICT", "CASE_RESOURCE_SCOPE_INACTIVE",
                            "CASE_RESOURCE_KNOWN_COLLECTION_MISSING", "CASE_RESOURCE_BINDING_CONFLICT"}:
                            code = "CASE_RESOURCE_CONFLICT"
                    raise CaseResourceClientError(code, retryable=response.status >= 500 or response.status == 429)
                result = CaseResourceResult.model_validate(await response.json())
                if result.request != request:
                    raise CaseResourceClientError("CASE_RESOURCE_RESPONSE_MISMATCH", retryable=False)
                return result
        except CaseResourceClientError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, AttributeError) as exc:
            raise CaseResourceClientError() from exc
