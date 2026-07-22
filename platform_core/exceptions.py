# exceptions.py
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse
from loguru import logger
from typing import NoReturn


# --------------------------------------------------
# 1. 基础设施层异常（DAO 层使用）
# --------------------------------------------------
class DatabaseException(Exception):
    """数据库异常 — 由全局 handler 统一转为 500"""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error


# --------------------------------------------------
# 2. 业务层异常（Service 层使用，均继承 HTTPException）
# --------------------------------------------------
class APIException(HTTPException):
    """通用 API 异常基类 — 支持动态 message 格式化"""

    def __init__(
        self,
        code: str,
        message: str,
        http_status: int = 400,
        detail: dict | None = None,
    ):
        formatted_message = message
        if detail:
            try:
                formatted_message = message.format(**detail)
            except Exception:
                pass

        super().__init__(
            status_code=http_status,
            detail={
                "code": code,
                "message": formatted_message,
                "detail": detail or {},
            },
        )


class NotFoundError(APIException):
    """资源不存在异常 → 404 (通用业务层使用)"""

    def __init__(self, message: str, **extra_details):
        code = extra_details.pop("code", "NOT_FOUND")
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_404_NOT_FOUND,
            detail=extra_details or None,
        )


class DataNotFoundException(Exception):
    """数据不存在异常。

    关键设计：继承自 plain Exception 而非 HTTPException。
    - 在普通 API 请求中：由全局 data_not_found_handler 转为 404 JSON
    - 在 SSE 异步生成器中：作为普通异常安全传播，不会触发 Starlette 的
      HTTPException 特殊处理（该处理会终止流式响应）。
    """
    is_business_exception = True  # 避免 DB session 回滚时打印 ERROR 日志

    def __init__(self, message: str):
        self.message = message
        self.status_code = status.HTTP_404_NOT_FOUND
        super().__init__(message)


class DataConflictException(APIException):
    """数据冲突异常 → 409"""

    def __init__(self, message: str):
        super().__init__(
            code="CONFLICT",
            message=message,
            http_status=status.HTTP_409_CONFLICT,
        )


class ParamValueError(APIException):
    """参数错误异常 → 400"""

    def __init__(self, message: str, **extra_details):
        super().__init__(
            code="VALUE_ERROR",
            message=message,
            http_status=status.HTTP_400_BAD_REQUEST,
            detail=extra_details or None,
        )


class AuthorizationError(APIException):
    """授权错误异常 → 401"""

    def __init__(self, message: str, **extra_details):
        code = extra_details.pop("code", "UNAUTHORIZED")
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_401_UNAUTHORIZED,
            detail=extra_details or None,
        )


class PrivilegeError(APIException):
    """权限错误异常 → 403"""

    def __init__(self, message: str, **extra_details):
        code = extra_details.pop("code", "FORBIDDEN")
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_403_FORBIDDEN,
            detail=extra_details or None,
        )


class ConflictError(APIException):
    """冲突错误异常 → 409"""

    def __init__(self, message: str, **extra_details):
        code = extra_details.pop("code", "CONFLICT")
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_409_CONFLICT,
            detail=extra_details or None,
        )


class InternalServerError(APIException):
    """服务器内部错误 → 500"""

    def __init__(self, message: str = "服务器内部错误", **extra_details):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=extra_details or None,
        )


# --------------------------------------------------
# 3. 异常翻译工具函数
# --------------------------------------------------
def handle_exception(e: Exception, msg: str) -> NoReturn:
    """将基础设施层异常统一翻译为 HTTP 异常。

    保留此函数供 Service 层可选使用。由于 DataNotFoundException /
    DataConflictException 已直接继承 APIException，它们会自动被
    FastAPI 转为正确的 HTTP 响应，不再需要显式转换。
    """
    # 1. APIException / DataNotFoundException — 直接透传
    if isinstance(e, (APIException, DataNotFoundException)):
        raise e

    # 2. 数据库底层异常 → 500
    if isinstance(e, DatabaseException):
        logger.error(f"{msg} [DB_ERR]: {getattr(e, 'original_error', e)}")
        raise InternalServerError(f"{msg}: {e.message}")

    # 3. 未知异常 → 500
    logger.exception(f"{msg} [UNKNOWN_ERR]: {e}")
    raise InternalServerError(f"{msg}: 发生未知错误")


# --------------------------------------------------
# 4. FastAPI 全局异常处理器（在 cube_main.py 中注册）
# --------------------------------------------------
async def data_not_found_handler(request: Request, exc: DataNotFoundException):
    """DataNotFoundException → 404 JSON"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": "NOT_FOUND",
            "message": exc.message,
            "detail": {},
        },
    )


async def database_exception_handler(request: Request, exc: DatabaseException):
    """DatabaseException → 500 JSON（含日志）"""
    original = getattr(exc, 'original_error', None)
    logger.error(f"未捕获的数据库异常: {exc.message} | 原始错误: {original}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "INTERNAL_ERROR",
            "message": f"数据库错误: {exc.message}",
        },
    )


async def api_exception_handler(request: Request, exc: APIException):
    """APIException 通用处理器 — 确保统一的 JSON 格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail,
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """兜底处理器 — 未知异常 → 500"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
        },
    )
