"""
微服务内部通信安全模块。

为所有微服务提供统一的内网令牌校验机制。
从 NexusCube 项目移植而来，适配 kbot3 命名规范。

使用方式:
    # 在微服务的 lifespan 之前:
    from platform_core.platform.security import create_internal_auth_middleware
    app.middleware("http")(create_internal_auth_middleware())
"""

import os
from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger

# 环境变量名
INTERNAL_TOKEN_ENV = "KBOT_INTERNAL_SERVICE_TOKEN"
DEFAULT_DEV_TOKEN = "kbot_internal_service_token"

# 自定义 Header 名 (统一内部服务通信令牌)
INTERNAL_TOKEN_HEADER = "X-KBot-Internal-Token"

# 不需要鉴权的路径 (健康检查等)
PUBLIC_PATHS = {"/health", "/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"}


def _mask_token(token: str) -> str:
    """对令牌进行脱敏显示（仅显示前 8 位）"""
    if len(token) <= 8:
        return token[:4] + "****"
    return token[:8] + "****"


def get_internal_token() -> str:
    """获取配置的内部通信令牌"""
    token = os.getenv(INTERNAL_TOKEN_ENV, DEFAULT_DEV_TOKEN)
    logger.debug(f"[InternalAuth] 服务端期望令牌: {_mask_token(token)}")
    return token


def create_internal_auth_middleware(
    token: str | None = None,
    public_paths: set[str] | None = None
):
    """
    创建内部认证中间件。

    自动放行健康检查等公开路径，其余请求需携带合法的 X-KBot-Internal-Token 头。

    Args:
        token: 期望的令牌值。不传则从环境变量 KBOT_INTERNAL_SERVICE_TOKEN 读取
        public_paths: 公开路径集合。不传则使用默认的 {/health, /docs, /redoc, /openapi.json}

    Returns:
        ASGI middleware 函数

    Example:
        from platform_core.platform.security import create_internal_auth_middleware
        app.middleware("http")(create_internal_auth_middleware())
    """
    expected_token = token or get_internal_token()
    skip_paths = public_paths or PUBLIC_PATHS

    # 开发环境下打印令牌提示
    if expected_token == DEFAULT_DEV_TOKEN:
        logger.warning(
            f"[Security] 使用默认开发令牌。生产环境请设置 {INTERNAL_TOKEN_ENV} 环境变量!"
        )

    async def middleware(request: Request, call_next):
        # 放行公开路径
        if request.url.path in skip_paths:
            return await call_next(request)

        # 放行 OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # 校验令牌
        provided_token = request.headers.get(INTERNAL_TOKEN_HEADER)
        if not provided_token:
            logger.warning(f"[Security] 拒绝无令牌请求: {request.method} {request.url.path} from {request.client}")
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Forbidden: 内部服务调用缺少认证令牌。"
                }
            )

        if provided_token != expected_token:
            logger.warning(
                f"[InternalAuth] 令牌不匹配: 期望={_mask_token(expected_token)}, 收到={_mask_token(provided_token)}, "
                f"请求={request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Forbidden: 内部服务令牌无效，拒绝访问。"
                }
            )

        return await call_next(request)

    return middleware
