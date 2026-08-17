"""Main API/BFF 的 FastAPI 应用工厂。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_offline import FastAPIOffline
from loguru import logger
from sqlalchemy import text

from main_api.api import (
    access_management_router,
    aiops_app_router,
    auth_router,
    conversation_router,
    data_query_router,
    development_agent_runs_router,
    development_logs_router,
    dify_router,
    domain_router,
    integration_router,
    knowledge_router,
    knowledge_retrieval_app_router,
    km_asset_app_router,
    memory_router,
    model_catalog_router,
    notification_router,
    ops_router,
    run_router,
    slack_router,
)
from main_api.config import get_main_api_settings
from main_api.application import UserAuthenticationError
from main_api.log_reader import LocalLogSearchService
from platform_clients import (
    AIOpsClientError,
    AgentRuntimeClientError,
    KnowledgeCoreClientError,
    KnowledgeRetrievalAppClientError,
    KmAssetClientError,
    DataQueryClientError,
)
from platform_core.middleware.log_middleware import log_requests
from platform_core.security import (
    PortalApiKeyError,
    PortalApiKeyVerifier,
    create_public_auth_middleware,
)


LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[Any]]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _repository_path(value: str) -> Path:
    """将部署配置中的相对路径固定解析到 KBot 仓库根目录。"""
    path = Path(value)
    return path if path.is_absolute() else _REPOSITORY_ROOT / path


def _log_downstream_failure(
    *,
    request: Request,
    service_name: str,
    exc: Exception,
) -> None:
    """记录已被异常处理器转换为公开响应的下游失败。"""
    context = getattr(request.state, "auth_context", None)
    request_id = (
        request.headers.get("X-Request-ID")
        or getattr(context, "request_id", None)
        or "-"
    )
    status_code = getattr(exc, "status_code", 500)
    error_code = getattr(exc, "code", type(exc).__name__)
    cause = exc.__cause__
    log_method = logger.error if status_code >= 500 else logger.warning
    log_method(
        "下游服务请求失败 | service={} | status_code={} | error_code={} "
        "| request_id={} | cause_type={} | cause={}",
        service_name,
        status_code,
        error_code,
        request_id,
        type(cause).__name__ if cause else type(exc).__name__,
        str(cause or exc),
    )


def _problem_response(
    *,
    request: Request,
    status_code: int,
    code: str,
    title: str,
    detail: str,
    field_errors: list[dict[str, Any]] | None = None,
) -> JSONResponse:
    context = getattr(request.state, "auth_context", None)
    request_id = (
        request.headers.get("X-Request-ID")
        or getattr(context, "request_id", None)
        or str(uuid4())
    )
    payload: dict[str, Any] = {
        "type": f"urn:kbot:error:{code.lower()}",
        "title": title,
        "status": status_code,
        "code": code,
        "detail": detail,
        "request_id": request_id,
    }
    if field_errors:
        payload["field_errors"] = field_errors
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        headers={"X-Request-ID": request_id},
        content=payload,
    )


def create_main_api_app(
    *,
    lifespan: LifespanFactory | None = None,
    verifier: PortalApiKeyVerifier | None = None,
    domain_validator=None,
    enable_access_log: bool = True,
) -> FastAPI:
    """构造只发布公开契约的 Main API 应用。"""
    settings = get_main_api_settings()
    config = settings.api
    if config.test_auth_bypass_enabled and not settings.is_development():
        raise RuntimeError("测试认证绕过只允许在 development 环境启用")
    if config.test_auth_bypass_enabled:
        logger.warning(
            "Main API 测试认证绕过已启用；仅供本地测试页面使用"
        )
    app_kwargs: dict[str, Any] = {}
    if lifespan is not None:
        app_kwargs["lifespan"] = lifespan
    app = FastAPIOffline(
        title="KBot Main API",
        description="KBot 4.0 的唯一公开 API/BFF 入口。",
        version=config.service_version,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url="/redoc" if config.docs_enabled else None,
        **app_kwargs,
    )
    app.state.service_name = config.service_name

    async def validate_domain(domain_id: str) -> bool:
        if domain_validator is not None:
            return await domain_validator(domain_id)
        service = getattr(app.state, "domain_validation_service", None)
        if service is None:
            return False
        return await service.is_active(domain_id)

    async def authenticate_user(request: Request):
        """接受由 Main API 签发、绑定用户与 Domain 的公开用户 Token。"""
        service = getattr(app.state, "user_auth_service", None)
        if service is None:
            return None
        context = service.authenticate_request(
            request.headers.get("Authorization")
        )
        if context is None:
            return None
        claims = service.verify(request.headers.get("Authorization"))
        try:
            await service.validate_session(claims=claims)
        except UserAuthenticationError as exc:
            raise PortalApiKeyError(exc.code, str(exc)) from exc
        if claims.must_change_password and request.url.path not in {
                "/api/v1/auth/me",
                "/api/v1/auth/password",
            "/api/v1/apps/km-asset/auth/password",
        }:
            raise PortalApiKeyError(
                "PASSWORD_CHANGE_REQUIRED", "首次登录必须先修改密码"
            )
        request.state.user_token_claims = claims
        return context

    if enable_access_log:
        app.middleware("http")(log_requests)
    domainless_paths = {
        "/api/v1/domains",
        "/api/v1/model-catalog",
    }
    if settings.platform.debug:
        domainless_paths.update(
            {
                "/api/v1/development/logs/services",
                "/api/v1/development/logs/events",
            }
        )
    app.middleware("http")(
        create_public_auth_middleware(
            verifier=verifier,
            domain_validator=validate_domain,
            allow_test_bypass=config.test_auth_bypass_enabled,
            alternate_authenticator=authenticate_user,
            public_paths={
                "/health",
                "/healthz",
                "/readyz",
                "/live",
                "/ready",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/api/v1/auth/platform/login",
                "/api/v1/auth/apps",
                "/api/v1/apps/km-asset/auth/login",
            },
            domainless_paths=domainless_paths,
            domainless_prefixes={"/api/v1/platform/"},
            public_prefixes={
                "/api/v1/auth/apps/",
                "/api/v1/integrations/monitoring/",
                "/api/v1/integrations/slack/",
                "/static-offline-docs/",
            },
        )
    )
    # CORS 必须位于最外层，确保认证及业务错误响应也携带跨域响应头。
    if config.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Idempotency-Key",
                "If-Match",
                "Last-Event-ID",
                "X-KBot-Domain-ID",
                "X-KBot-Test-Auth",
                "X-KBot-User-ID",
                "X-Request-ID",
                "traceparent",
            ],
        )
    app.include_router(knowledge_router)
    app.include_router(auth_router)
    app.include_router(access_management_router)
    app.include_router(knowledge_retrieval_app_router)
    app.include_router(km_asset_app_router)
    app.include_router(aiops_app_router)
    app.include_router(model_catalog_router)
    app.include_router(notification_router)
    app.include_router(domain_router)
    app.include_router(run_router)
    app.include_router(conversation_router)
    app.include_router(data_query_router)
    app.include_router(dify_router)
    app.include_router(memory_router)
    app.include_router(ops_router)
    app.include_router(integration_router)
    app.include_router(slack_router)
    if settings.platform.debug:
        log_config = settings.development_logs
        app.state.development_log_root = _repository_path(settings.log.dir)
        app.state.development_log_search_service = LocalLogSearchService(
            log_root=app.state.development_log_root,
            topology_path=_repository_path(log_config.topology_path),
            max_files_per_stream=log_config.max_files_per_stream,
            max_bytes_per_file=log_config.max_bytes_per_file,
            max_total_scan_bytes=log_config.max_total_scan_bytes,
            max_window_hours=log_config.max_window_hours,
            max_page_size=log_config.max_page_size,
            max_export_events=log_config.max_export_events,
            max_detail_chars=log_config.max_detail_chars,
            max_field_chars=log_config.max_field_chars,
        )
        app.include_router(development_logs_router)
        app.include_router(development_agent_runs_router)

    @app.exception_handler(UserAuthenticationError)
    async def user_authentication_error_handler(
        request: Request, exc: UserAuthenticationError,
    ):
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            title=(
                "系统尚未初始化"
                if exc.code == "SYSTEM_NOT_INITIALIZED"
                else "用户认证失败"
            ),
            detail=str(exc),
        )

    @app.exception_handler(KnowledgeCoreClientError)
    async def knowledge_core_error_handler(
        request: Request,
        exc: KnowledgeCoreClientError,
    ):
        _log_downstream_failure(
            request=request,
            service_name="knowledge-core",
            exc=exc,
        )
        if exc.status_code in {401, 403}:
            return _problem_response(
                request=request,
                status_code=502,
                code="UPSTREAM_AUTH_FAILED",
                title="下游服务认证失败",
                detail="Knowledge Core 内部认证失败",
            )
        if exc.status_code >= 500:
            return _problem_response(
                request=request,
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                title="Knowledge Core 暂时不可用",
                detail="Knowledge Core 暂时无法完成请求",
            )
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            title="Knowledge Core 请求失败",
            detail=str(exc),
        )

    @app.exception_handler(KnowledgeRetrievalAppClientError)
    async def knowledge_retrieval_app_error_handler(
        request: Request, exc: KnowledgeRetrievalAppClientError,
    ):
        _log_downstream_failure(
            request=request, service_name="knowledge-retrieval-app", exc=exc
        )
        if exc.status_code >= 500:
            return _problem_response(
                request=request, status_code=503,
                code="KNOWLEDGE_RETRIEVAL_APP_UNAVAILABLE",
                title="知识检索应用暂时不可用",
                detail="知识检索应用暂时无法完成请求",
            )
        return _problem_response(
            request=request, status_code=exc.status_code, code=exc.code,
            title="知识检索应用请求失败", detail=str(exc),
        )

    @app.exception_handler(KmAssetClientError)
    async def km_asset_app_error_handler(request: Request, exc: KmAssetClientError):
        _log_downstream_failure(request=request, service_name="km-asset-app", exc=exc)
        if exc.status_code >= 500:
            return _problem_response(request=request, status_code=503, code="KM_ASSET_APP_UNAVAILABLE", title="KM Asset 应用暂时不可用", detail="KM Asset 应用暂时无法完成请求")
        return _problem_response(request=request, status_code=exc.status_code, code=exc.code, title="KM Asset 应用请求失败", detail=str(exc))

    @app.exception_handler(DataQueryClientError)
    async def data_query_error_handler(
        request: Request,
        exc: DataQueryClientError,
    ):
        _log_downstream_failure(
            request=request,
            service_name="data-query",
            exc=exc,
        )
        if exc.status_code in {401, 403}:
            return _problem_response(
                request=request,
                status_code=502,
                code="UPSTREAM_AUTH_FAILED",
                title="下游服务认证失败",
                detail="Data Query 内部认证失败",
            )
        if exc.status_code >= 500:
            return _problem_response(
                request=request,
                status_code=503,
                code="DATA_QUERY_UNAVAILABLE",
                title="Data Query 暂时不可用",
                detail="Data Query 暂时无法完成请求",
            )
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            title="Data Query 请求失败",
            detail=str(exc),
        )

    @app.exception_handler(AgentRuntimeClientError)
    async def agent_runtime_error_handler(
        request: Request,
        exc: AgentRuntimeClientError,
    ):
        _log_downstream_failure(
            request=request,
            service_name="agent-runtime",
            exc=exc,
        )
        if exc.status_code in {401, 403}:
            return _problem_response(
                request=request,
                status_code=502,
                code="UPSTREAM_AUTH_FAILED",
                title="下游服务认证失败",
                detail="Agent Runtime 内部认证失败",
            )
        if exc.status_code >= 500:
            return _problem_response(
                request=request,
                status_code=503,
                code="AGENT_RUNTIME_UNAVAILABLE",
                title="Agent Runtime 暂时不可用",
                detail="Agent Runtime 暂时无法完成请求",
            )
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            title="Agent Runtime 请求失败",
            detail=str(exc),
        )

    @app.exception_handler(AIOpsClientError)
    async def aiops_error_handler(
        request: Request,
        exc: AIOpsClientError,
    ):
        _log_downstream_failure(
            request=request,
            service_name="aiops",
            exc=exc,
        )
        internal_auth_codes = {
            "AUTH_CONTEXT_REQUIRED",
            "AUTH_CONTEXT_EXPIRED",
            "INVALID_AUTH_CONTEXT",
            "SERVICE_IDENTITY_NOT_CONFIGURED",
            "SERVICE_IDENTITY_REQUIRED",
            "SERVICE_IDENTITY_EXPIRED",
            "INVALID_SERVICE_IDENTITY",
            "SERVICE_CALLER_DENIED",
            "SERVICE_SCOPE_DENIED",
            "SERVICE_CONTEXT_MISMATCH",
        }
        if exc.status_code == 401 or (
            exc.status_code == 403 and exc.code in internal_auth_codes
        ):
            return _problem_response(
                request=request,
                status_code=502,
                code="UPSTREAM_AUTH_FAILED",
                title="下游服务认证失败",
                detail="AIOps 内部认证失败",
            )
        if exc.status_code >= 500:
            return _problem_response(
                request=request,
                status_code=503,
                code="AIOPS_UNAVAILABLE",
                title="AIOps 暂时不可用",
                detail="AIOps 暂时无法完成请求",
            )
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            title="AIOps 请求失败",
            detail=str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        field_errors = [
            {
                "location": list(item.get("loc", ())),
                "message": item.get("msg", "字段无效"),
                "type": item.get("type", "value_error"),
            }
            for item in exc.errors()
        ]
        return _problem_response(
            request=request,
            status_code=422,
            code="REQUEST_VALIDATION_FAILED",
            title="请求字段校验失败",
            detail="一个或多个请求字段无效",
            field_errors=field_errors,
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            code = str(detail.get("code", "HTTP_ERROR"))
            message = str(detail.get("message", detail))
        else:
            code = "HTTP_ERROR"
            message = str(detail)
        return _problem_response(
            request=request,
            status_code=exc.status_code,
            code=code,
            title="请求处理失败",
            detail=message,
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception(
            "Main API 未处理异常：method={} path={} type={}",
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        response = _problem_response(
            request=request,
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            title="服务内部错误",
            detail="请求暂时无法完成",
        )
        origin = request.headers.get("Origin")
        if origin and "*" in config.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin.rstrip("/") in config.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        return response

    @app.get("/healthz", tags=["System"], summary="存活检查")
    async def healthz():
        return {
            "status": "ok",
            "service": config.service_name,
            "version": config.service_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/readyz", tags=["System"], summary="就绪检查")
    async def readyz(request: Request):
        checks: dict[str, str] = {}
        db_runtime = getattr(request.app.state, "db_runtime", None)
        if db_runtime is None:
            checks["database"] = "not_configured"
        else:
            try:
                async with db_runtime.session_factory() as session:
                    await session.execute(text("SELECT 1 FROM DUAL"))
                checks["database"] = "ok"
            except Exception:
                checks["database"] = "unavailable"
        kc_client = getattr(
            request.app.state,
            "knowledge_core_client",
            None,
        )
        if kc_client is None:
            checks["knowledge_core"] = "not_configured"
        else:
            checks["knowledge_core"] = (
                "ok" if await kc_client.is_ready() else "unavailable"
            )
        agent_client = getattr(
            request.app.state, "agent_runtime_client", None
        )
        if agent_client is None:
            checks["agent_runtime"] = "not_configured"
        else:
            checks["agent_runtime"] = (
                "ok" if await agent_client.is_ready() else "unavailable"
            )
        aiops_client = getattr(request.app.state, "aiops_client", None)
        if aiops_client is None:
            checks["aiops"] = "not_configured"
        else:
            checks["aiops"] = (
                "ok" if await aiops_client.is_ready() else "unavailable"
            )
        ready = all(value == "ok" for value in checks.values())
        if not ready:
            logger.warning("Main API 尚未就绪：checks={}", checks)
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "SERVICE_NOT_READY",
                    "message": "一个或多个内部依赖暂时不可用",
                },
            )
        return {"status": "ready"}

    return app
