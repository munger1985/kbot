"""Main API/BFF 的 FastAPI 应用工厂。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_offline import FastAPIOffline
from loguru import logger
from sqlalchemy import text

from main_api.api import knowledge_router
from platform_clients import KnowledgeCoreClientError
from platform_core.config.settings import get_app_config, get_main_api_config
from platform_core.middleware.log_middleware import log_requests
from platform_core.security import (
    PortalApiKeyVerifier,
    create_public_auth_middleware,
)


LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[Any]]


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
    config = get_main_api_config()
    app_kwargs: dict[str, Any] = {}
    if lifespan is not None:
        app_kwargs["lifespan"] = lifespan
    app = FastAPIOffline(
        title="KBot Main API",
        description="KBot 4.0 的唯一公开 API/BFF 入口。",
        version=config.service_version,
        docs_url="/docs" if get_app_config().debug else None,
        redoc_url="/redoc" if get_app_config().debug else None,
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
                "X-KBot-User-ID",
                "X-Request-ID",
                "traceparent",
            ],
        )
    if enable_access_log:
        app.middleware("http")(log_requests)
    app.middleware("http")(
        create_public_auth_middleware(
            verifier=verifier,
            domain_validator=validate_domain,
        )
    )
    app.include_router(knowledge_router)

    @app.exception_handler(KnowledgeCoreClientError)
    async def knowledge_core_error_handler(
        request: Request,
        exc: KnowledgeCoreClientError,
    ):
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
        return _problem_response(
            request=request,
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            title="服务内部错误",
            detail="请求暂时无法完成",
        )

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
