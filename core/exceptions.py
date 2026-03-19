# exceptions.py
from fastapi import HTTPException, status
from loguru import logger
from typing import NoReturn


def safe_log_error(msg: str, error: Exception, max_length: int = 1000) -> str:
    """
    安全地记录错误信息，限制错误字符串长度以避免打印大量数据（如向量）
    
    Args:
        msg: 错误消息前缀
        error: 异常对象
        max_length: 错误字符串最大长度，默认1000
        
    Returns:
        截断后的错误字符串
    """
    error_str = str(error)
    if len(error_str) > max_length:
        error_str = error_str[:max_length] + f"... (truncated, original length: {len(error_str)})"
    logger.error(f"{msg}: {error_str}")
    return error_str

# --------------------------------------------------
# 1. Infrastructure layer exceptions (used in DAO layer)
# --------------------------------------------------
class DatabaseException(Exception):
    """Database exception"""
    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error
        if original_error:
            # 限制原始异常的字符串长度，避免打印大量向量数据
            error_str = str(original_error)
            if len(error_str) > 500:
                error_str = error_str[:500] + "... (truncated)"
            logger.debug(f"[DatabaseException] Created: {message}, original error type: {type(original_error).__name__}, "
                        f"original error: {error_str}")

class DataNotFoundException(DatabaseException):
    """Data not found exception"""
    def __init__(self, message: str):
        super().__init__(message)

class DataConflictException(DatabaseException):
    """Data conflict exception"""
    def __init__(self, message: str):
        super().__init__(message)

# --------------------------------------------------
# 2. Business layer exceptions (used in Service layer)
# --------------------------------------------------
class APIException(HTTPException):
    """Generic API exception base class - supports dynamic message"""
    def __init__(
        self,
        code: str,                    # Business error code
        message: str,                 # Dynamic message template
        http_status: int = 400,       # HTTP status code
        detail: dict | None = None    # Additional information (can be used for message formatting)
    ):
        # If there is data to format in detail
        formatted_message = message
        if detail:
            # Simple formatting: {key} in message will be replaced with detail[key]
            try:
                formatted_message = message.format(**detail)
            except:
                pass  # Use original message if formatting fails
        
        super().__init__(
            status_code=http_status,
            detail={
                "code": code,
                "message": formatted_message,  # ✅ Formatted message
                "detail": detail or {}
            }
        )

class NotFoundError(APIException):
    """Resource not found exception"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: Specific description of resource not found
            **extra_details: Additional information (e.g., resource_type, resource_id, etc.)
        """
        # Use generic NOT_FOUND code if not specified
        code = extra_details.pop("code", "NOT_FOUND")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_404_NOT_FOUND,
            detail=extra_details or None
        )

class ParamValueError(APIException):
    """Parameter value error exception"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: Message template, e.g., "The selected {param} value is invalid"
            **extra_details: Formatting parameters, e.g., param="color"
        """
        super().__init__(
            code="VALUE_ERROR",
            message=message,
            http_status=status.HTTP_400_BAD_REQUEST,
            detail=extra_details or None
        )

class AuthorizationError(APIException):
    """Authorization error exception"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: Specific description of authorization failure
            **extra_details: Additional information (e.g., required_role, current_role, etc.)
        """
        # Use generic UNAUTHORIZED code if not specified
        code = extra_details.pop("code", "UNAUTHORIZED")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_401_UNAUTHORIZED,
            detail=extra_details or None
        )

class PrivilegeError(APIException):
    """Privilege error exception"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: Specific description of insufficient privileges
            **extra_details: Additional information (e.g., required_privilege, current_privilege, etc.)
        """
        # Use generic FORBIDDEN code if not specified
        code = extra_details.pop("code", "FORBIDDEN")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_403_FORBIDDEN,
            detail=extra_details or None
        )

class InternalServerError(APIException):
    """Internal server error"""
    def __init__(self, message: str = "Internal server error", **extra_details):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=extra_details or None
        )


def handle_exception(e: Exception, msg: str) -> NoReturn:
    """Exception management standardization"""
    if isinstance(e, DataNotFoundException):
        raise NotFoundError(e.message)
    if isinstance(e, DataConflictException):
        raise ParamValueError(e.message)
    if isinstance(e, (DatabaseException)):
        # 输出详细错误日志，包括原始异常的堆栈信息
        logger.error(f"{msg}: {e.message}")
        if e.original_error:
            logger.exception(f"原始异常详情: {type(e.original_error).__name__}", exc_info=e.original_error)
        raise InternalServerError(f"{msg}: {e.message}")
    if isinstance(e, (NotFoundError, ParamValueError, AuthorizationError, PrivilegeError, InternalServerError)):
        raise e
    logger.exception(f"{msg}: {e}")
    raise e