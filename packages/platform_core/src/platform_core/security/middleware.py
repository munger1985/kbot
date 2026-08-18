"""FastAPI 的公开入口和内部服务认证中间件。"""

from __future__ import annotations

import hmac
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from platform_core.contracts import AuthContext, PrincipalKind, PUBLIC_API_V1

from .api_key import PortalApiKeyError, PortalApiKeyVerifier
from .auth_context import (
    AUTH_CONTEXT_HEADER,
    AuthContextJWTCodec,
    AuthContextTokenError,
)
from .runtime import (
    INTERNAL_TOKEN_HEADER,
    create_auth_context_codec,
    create_model_api_key_verifier,
    get_internal_service_token,
)
from .service_identity import (
    SERVICE_IDENTITY_HEADER,
    ServiceIdentityJWTCodec,
    ServiceIdentityTokenError,
)


DOMAIN_ID_HEADER = "X-KBot-Domain-ID"
TENANT_ID_HEADER = "X-KBot-Tenant-ID"
USER_ID_HEADER = "X-KBot-User-ID"
TEST_AUTH_BYPASS_HEADER = "X-KBot-Test-Auth"
PUBLIC_PATHS = {
    "/health",
    "/healthz",
    "/readyz",
    "/live",
    "/ready",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
}
DomainValidator = Callable[[str], Awaitable[bool]]
TenantValidator = Callable[[str], Awaitable[bool]]
AlternateAuthenticator = Callable[[Request], Awaitable[AuthContext | None]]


def _problem(
    *,
    request: Request,
    status_code: int,
    code: str,
    detail: str,
) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    headers = {"X-Request-ID": request_id}
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        headers=headers,
        content={
            "type": f"urn:kbot:error:{code.lower()}",
            "title": "请求认证失败",
            "status": status_code,
            "code": code,
            "detail": detail,
            "request_id": request_id,
        },
    )


def _api_client_problem(
    *, status_code: int, code: str, detail: str,
) -> JSONResponse:
    """返回 OpenAI SDK 可识别的认证错误结构。"""
    return JSONResponse(
        status_code=status_code,
        headers={"WWW-Authenticate": "Bearer"},
        content={
            "error": {
                "message": detail,
                "type": "authentication_error",
                "param": None,
                "code": code,
            }
        },
    )


def _request_ids(request: Request) -> tuple[str, str]:
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    traceparent = request.headers.get("traceparent")
    trace_id = traceparent or request_id
    return request_id[:128], trace_id[:128]


def _validated_header(
    request: Request,
    *,
    name: str,
    required: bool,
    max_length: int,
) -> str | None:
    value = request.headers.get(name)
    if value is None:
        if required:
            raise PortalApiKeyError(
                "IDENTITY_CONTEXT_REQUIRED",
                f"缺少 {name} Header",
            )
        return None
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise PortalApiKeyError(
            "INVALID_IDENTITY_CONTEXT",
            f"{name} Header 格式无效",
        )
    return normalized


def create_public_auth_middleware(
    *,
    domain_validator: DomainValidator,
    tenant_validator: TenantValidator | None = None,
    public_paths: set[str] | None = None,
    public_prefixes: set[str] | None = None,
    domainless_paths: set[str] | None = None,
    domainless_prefixes: set[str] | None = None,
    allow_test_bypass: bool = False,
    alternate_authenticator: AlternateAuthenticator | None = None,
):
    """创建 Main API 的用户 Token、App Key 与开发绕过认证中间件。"""
    skip_paths = PUBLIC_PATHS if public_paths is None else public_paths

    async def middleware(request: Request, call_next):
        if (
            request.url.path in skip_paths
            or any(
                request.url.path.startswith(prefix)
                for prefix in (public_prefixes or set())
            )
            or request.method == "OPTIONS"
        ):
            return await call_next(request)
        try:
            domainless_request = (
                request.url.path in (domainless_paths or set())
                or any(
                    request.url.path.startswith(prefix)
                    for prefix in (domainless_prefixes or set())
                )
            )
            test_bypass = (
                allow_test_bypass
                and request.headers.get(TEST_AUTH_BYPASS_HEADER, "").lower()
                == "true"
            )
            alternate_context = None
            if not test_bypass and alternate_authenticator is not None:
                alternate_context = await alternate_authenticator(request)
            if alternate_context is not None:
                domain_id = alternate_context.domain_id
                user_id = alternate_context.asserted_user_id
                platform_entry = str(alternate_context.entry_kind) == "PLATFORM"
                if not user_id or (not platform_entry and not domain_id):
                    raise PortalApiKeyError(
                        "INVALID_IDENTITY_CONTEXT",
                        "备用认证没有提供完整的平台或业务身份上下文",
                    )
                if domain_id:
                    try:
                        domain_is_active = await domain_validator(domain_id)
                    except Exception as exc:
                        logger.error(
                            "Domain 校验依赖不可用：method={} path={} type={}",
                            request.method,
                            request.url.path,
                            type(exc).__name__,
                        )
                        return _problem(
                            request=request,
                            status_code=503,
                            code="IDENTITY_SERVICE_UNAVAILABLE",
                            detail="身份上下文暂时无法校验",
                        )
                    if not domain_is_active:
                        raise PortalApiKeyError("INVALID_DOMAIN", "Domain 不存在或已停用")
                request.state.auth_context = alternate_context
                response = await call_next(request)
                response.headers.setdefault(
                    "X-Request-ID", alternate_context.request_id
                )
                return response
            if test_bypass:
                principal_client_id = "kbot-development-test"
                principal_key_id = "development-test-bypass"
            else:
                raise PortalApiKeyError(
                    "AUTH_REQUIRED",
                    "需要用户 Token 或绑定当前 App 的 API Key",
                )
            domain_id = None
            if not domainless_request:
                domain_id = _validated_header(
                    request,
                    name=DOMAIN_ID_HEADER,
                    required=True,
                    max_length=128,
                )
            user_id = _validated_header(
                request,
                name=USER_ID_HEADER,
                required=True,
                max_length=256,
            )
            if domain_id:
                try:
                    domain_is_active = await domain_validator(domain_id)
                except Exception as exc:
                    logger.error(
                        "Domain 校验依赖不可用：method={} path={} type={}",
                        request.method,
                        request.url.path,
                        type(exc).__name__,
                    )
                    return _problem(
                        request=request,
                        status_code=503,
                        code="IDENTITY_SERVICE_UNAVAILABLE",
                        detail="身份上下文暂时无法校验",
                    )
                if not domain_is_active:
                    raise PortalApiKeyError(
                        "INVALID_DOMAIN",
                        "Domain 不存在或已停用",
                    )
            tenant_id = None
            if not domainless_request:
                tenant_id = _validated_header(
                    request,
                    name=TENANT_ID_HEADER,
                    required=False,
                    max_length=128,
                )
            if tenant_id and tenant_validator is not None:
                try:
                    tenant_is_valid = await tenant_validator(tenant_id)
                except Exception as exc:
                    logger.error(
                        "Tenant 校验依赖不可用：method={} path={} type={}",
                        request.method,
                        request.url.path,
                        type(exc).__name__,
                    )
                    return _problem(
                        request=request,
                        status_code=503,
                        code="IDENTITY_SERVICE_UNAVAILABLE",
                        detail="身份上下文暂时无法校验",
                    )
                if not tenant_is_valid:
                    raise PortalApiKeyError(
                        "INVALID_TENANT",
                        "Tenant 不存在或已停用",
                    )
            request_id, trace_id = _request_ids(request)
            request.state.auth_context = AuthContext(
                principal_kind=(
                    PrincipalKind.API_CLIENT
                    if domainless_request
                    else PrincipalKind.PORTAL
                ),
                client_id=principal_client_id,
                api_key_id=principal_key_id,
                domain_id=domain_id,
                tenant_id=tenant_id,
                asserted_user_id=user_id,
                request_id=request_id,
                trace_id=trace_id,
            )
            response = await call_next(request)
            response.headers.setdefault("X-Request-ID", request_id)
            return response
        except PortalApiKeyError as exc:
            logger.warning(
                "拒绝公开 API 请求：code={} method={} path={}",
                exc.code,
                request.method,
                request.url.path,
            )
            if exc.code == "APP_API_KEY_RATE_LIMITED":
                status_code = 429
            elif exc.code == "APP_API_KEY_IDENTITY_HEADER_FORBIDDEN":
                status_code = 400
            elif exc.code == "APP_API_KEY_EXPIRED":
                status_code = 401
            elif exc.code.startswith("APP_API_KEY_"):
                status_code = 403
            else:
                status_code = 401 if exc.code in {
                    "AUTH_REQUIRED",
                    "INVALID_AUTH_SCHEME",
                    "INVALID_API_KEY",
                    "API_KEY_DISABLED",
                    "API_KEY_EXPIRED",
                    "INVALID_USER_TOKEN",
                    "INVALID_APP_API_KEY",
                    "APP_API_CLIENT_DISABLED",
                    "USER_TOKEN_EXPIRED",
                    "USER_DISABLED",
                    "USER_SESSION_REVOKED",
                    "DOMAIN_ACCESS_DENIED",
                    "PASSWORD_CHANGE_REQUIRED",
                } else 400
            return _problem(
                request=request,
                status_code=status_code,
                code=exc.code,
                detail=str(exc),
            )

    return middleware


def create_api_client_auth_middleware(
    *,
    verifier: PortalApiKeyVerifier | None = None,
    api_prefix: str = PUBLIC_API_V1,
):
    """认证不携带 Domain 的标准 API Client，例如模型推理调用方。"""
    resolved_verifier = verifier or create_model_api_key_verifier()

    async def middleware(request: Request, call_next):
        if (
            request.method == "OPTIONS"
            or request.url.path in PUBLIC_PATHS
            or not request.url.path.startswith(api_prefix)
        ):
            return await call_next(request)
        try:
            principal = resolved_verifier.verify_authorization(
                request.headers.get("Authorization")
            )
            request_id, trace_id = _request_ids(request)
            request.state.auth_context = AuthContext(
                principal_kind=PrincipalKind.API_CLIENT,
                client_id=principal.client_id,
                api_key_id=principal.key_id,
                request_id=request_id,
                trace_id=trace_id,
            )
            response = await call_next(request)
            response.headers.setdefault("X-Request-ID", request_id)
            return response
        except PortalApiKeyError as exc:
            logger.warning(
                "拒绝 API Client 请求：code={} method={} path={}",
                exc.code,
                request.method,
                request.url.path,
            )
            return _api_client_problem(
                status_code=401,
                code=exc.code,
                detail=str(exc),
            )

    return middleware


def create_internal_auth_middleware(
    *,
    audience: str,
    codec: AuthContextJWTCodec | None = None,
    service_token: str | None = None,
    public_paths: set[str] | None = None,
    skip_prefixes: tuple[str, ...] = (),
):
    """创建内部服务使用的服务凭证与 AuthContext JWT 双重认证。"""
    if not audience:
        raise ValueError("内部认证 audience 不能为空")
    resolved_codec = codec or create_auth_context_codec()
    expected_service_token = service_token or get_internal_service_token()
    skip_paths = PUBLIC_PATHS if public_paths is None else public_paths

    async def middleware(request: Request, call_next):
        if (
            request.url.path in skip_paths
            or request.method == "OPTIONS"
            or request.url.path.startswith(skip_prefixes)
        ):
            return await call_next(request)
        provided_service_token = request.headers.get(INTERNAL_TOKEN_HEADER)
        if (
            not provided_service_token
            or not hmac.compare_digest(
                provided_service_token,
                expected_service_token,
            )
        ):
            logger.warning(
                "拒绝内部请求：服务凭证无效 method={} path={}",
                request.method,
                request.url.path,
            )
            return _problem(
                request=request,
                status_code=401,
                code="INVALID_SERVICE_CREDENTIAL",
                detail="内部服务凭证无效",
            )
        try:
            context = resolved_codec.verify(
                request.headers.get(AUTH_CONTEXT_HEADER, ""),
                audience=audience,
            )
        except AuthContextTokenError as exc:
            logger.warning(
                "拒绝内部请求：code={} method={} path={}",
                exc.code,
                request.method,
                request.url.path,
            )
            return _problem(
                request=request,
                status_code=401,
                code=exc.code,
                detail=str(exc),
            )
        request.state.auth_context = context
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", context.request_id)
        return response

    return middleware


def create_scoped_internal_auth_middleware(
    *,
    audience: str,
    allowed_callers: dict[str, frozenset[str]],
    service_identity_codec: ServiceIdentityJWTCodec | None = None,
    auth_context_codec: AuthContextJWTCodec | None = None,
    public_paths: set[str] | None = None,
):
    """创建 AIOps 使用的双 JWT、调用方和 scope 认证中间件。"""
    if not audience:
        raise ValueError("内部认证 audience 不能为空")
    skip_paths = PUBLIC_PATHS if public_paths is None else public_paths

    async def middleware(request: Request, call_next):
        if request.url.path in skip_paths or request.method == "OPTIONS":
            return await call_next(request)
        try:
            resolved_identity_codec = service_identity_codec or getattr(
                request.app.state,
                "service_identity_codec",
                None,
            )
            resolved_auth_codec = auth_context_codec or getattr(
                request.app.state,
                "auth_context_codec",
                None,
            )
            if resolved_identity_codec is None or resolved_auth_codec is None:
                raise ServiceIdentityTokenError(
                    "SERVICE_IDENTITY_NOT_CONFIGURED",
                    "内部身份验证器尚未初始化",
                )
            identity = resolved_identity_codec.verify(
                request.headers.get(SERVICE_IDENTITY_HEADER, ""),
                audience=audience,
            )
            allowed_scopes = allowed_callers.get(identity.subject)
            if allowed_scopes is None:
                raise ServiceIdentityTokenError(
                    "SERVICE_CALLER_DENIED",
                    "内部调用方未获准访问当前服务",
                )
            presented_scopes = frozenset(identity.scopes)
            if not presented_scopes.issubset(allowed_scopes):
                raise ServiceIdentityTokenError(
                    "SERVICE_SCOPE_DENIED",
                    "Service Identity 包含未授权 scope",
                )
            context = resolved_auth_codec.verify(
                request.headers.get(AUTH_CONTEXT_HEADER, ""),
                audience=audience,
            )
            if context.calling_service != identity.subject:
                raise ServiceIdentityTokenError(
                    "SERVICE_CONTEXT_MISMATCH",
                    "Service Identity 与 AuthContext 调用方不一致",
                )
        except (ServiceIdentityTokenError, AuthContextTokenError) as exc:
            logger.warning(
                "拒绝 AIOps 内部请求：code={} method={} path={}",
                exc.code,
                request.method,
                request.url.path,
            )
            return _problem(
                request=request,
                status_code=403,
                code=exc.code,
                detail=str(exc),
            )

        request.state.service_identity = identity
        request.state.auth_context = context
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", context.request_id)
        return response

    return middleware


def get_auth_context(request: Request) -> AuthContext:
    """从已认证请求中读取 AuthContext。"""
    context: Any = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext):
        raise RuntimeError("请求尚未通过身份认证")
    return context


def get_actor_id(request: Request) -> str:
    """从可信 AuthContext 派生审计操作人，忽略客户端 Actor Header。"""
    context = get_auth_context(request)
    if context.asserted_user_id:
        return f"user:{context.asserted_user_id}"
    return f"svc:{context.client_id}"


def require_domain_match(request: Request, domain_id: str | int) -> None:
    """当身份上下文携带 Domain 时，强制与目标资源 Domain 一致。"""
    context = get_auth_context(request)
    if context.domain_id is None:
        return
    if context.domain_id != str(domain_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "RESOURCE_NOT_FOUND",
                "message": "资源不存在",
            },
        )
