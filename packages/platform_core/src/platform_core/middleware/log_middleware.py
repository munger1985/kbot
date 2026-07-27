"""HTTP Access 日志中间件。"""

from __future__ import annotations

import time

from fastapi import Request
from loguru import logger

from platform_core.config.settings import get_log_config


_IGNORED_PATHS = {"/docs", "/redoc", "/openapi.json"}
_QUIET_PATHS = {
    "/health",
    "/healthz",
    "/live",
    "/ready",
    "/readyz",
    "/metrics",
    "/internal/v1/knowledge/parse-tasks/claim",
    "/internal/v1/knowledge/projection-tasks/claim",
}


def _safe_url(request: Request) -> str:
    path = request.url.path
    if path.startswith("/api/v1/integrations/monitoring/"):
        return "/api/v1/integrations/monitoring/{redacted}/events"
    query = request.url.query
    return f"{path}?{query}" if query else path


def _access_level(status_code: int, *, quiet: bool) -> str:
    if status_code >= 500:
        return "ERROR"
    if status_code >= 400:
        return "WARNING"
    return "DEBUG" if quiet else "INFO"


def _write_access(
    *,
    level: str,
    method: str,
    url: str,
    status_code: int,
    duration_ms: float,
    client_host: str,
    request_id: str | None,
) -> None:
    message = (
        "API 访问 | method={} | path={} | status={} | "
        "duration_ms={:.2f} | client={} | request_id={}"
    )
    logger.bind(log_type="access").log(
        level,
        message,
        method,
        url,
        status_code,
        duration_ms,
        client_host,
        request_id or "-",
    )


async def log_requests(request: Request, call_next):
    """每个 HTTP 请求只写一条完成态 Access 日志。"""

    settings = get_log_config()
    path = request.url.path
    ignored = (
        path in _IGNORED_PATHS
        or path.startswith("/api/v1/development/logs")
    )
    quiet = path in _QUIET_PATHS
    method = request.method
    url = _safe_url(request)
    client_host = request.client.host if request.client else "unknown"
    request_id = request.headers.get("X-Request-ID")
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - started_at) * 1000
        if not ignored:
            logger.exception(
                "API 未处理异常 | method={} | path={} | "
                "exception_type={} | 错误={} | duration_ms={:.2f}",
                method,
                url,
                type(exc).__name__,
                str(exc),
                duration_ms,
            )
            _write_access(
                level="ERROR",
                method=method,
                url=url,
                status_code=500,
                duration_ms=duration_ms,
                client_host=client_host,
                request_id=request_id,
            )
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    if settings.api_log_enabled and not ignored:
        _write_access(
            level=_access_level(response.status_code, quiet=quiet),
            method=method,
            url=url,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client_host=client_host,
            request_id=request_id,
        )
    response.headers["X-Process-Time"] = str(duration_ms)
    return response
