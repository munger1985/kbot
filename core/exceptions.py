# exceptions.py
from fastapi import HTTPException, status
from loguru import logger
from typing import NoReturn

# --------------------------------------------------
# 1. 基础设施层异常（DAO层使用）
# --------------------------------------------------
class DatabaseException(Exception):
    """数据库异常"""
    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error

class DataNotFoundException(DatabaseException):
    """数据不存在异常"""
    def __init__(self, message: str):
        super().__init__(message)

class DataConflictException(DatabaseException):
    """数据冲突异常"""
    def __init__(self, message: str):
        super().__init__(message)

# --------------------------------------------------
# 2. 业务层异常（Service层使用）
# --------------------------------------------------
class APIException(HTTPException):
    """通用API异常基类 - 支持动态message"""
    def __init__(
        self,
        code: str,                    # 业务错误码
        message: str,                 # 动态消息模板
        http_status: int = 400,       # HTTP状态码
        detail: dict | None = None    # 额外信息（可用于message格式化）
    ):
        # 如果detail中有需要格式化的数据
        formatted_message = message
        if detail:
            # 简单格式化：message中的 {key} 会被替换为 detail[key]
            try:
                formatted_message = message.format(**detail)
            except:
                pass  # 格式化失败，使用原消息
        
        super().__init__(
            status_code=http_status,
            detail={
                "code": code,
                "message": formatted_message,  # ✅ 格式化后的消息
                "detail": detail or {}
            }
        )

class NotFoundError(APIException):
    """资源不存在异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 资源不存在的具体描述
            **extra_details: 额外信息（如resource_type、resource_id等）
        """
        # 如果没有指定code，默认使用通用的NOT_FOUND
        code = extra_details.pop("code", "NOT_FOUND")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_404_NOT_FOUND,
            detail=extra_details or None
        )

class ParamValueError(APIException):
    """参数错误异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 消息模板，如"选择的{param}值无效"
            **kwargs: 格式化参数，如param="颜色"
        """
        super().__init__(
            code="VALUE_ERROR",
            message=message,
            http_status=status.HTTP_400_BAD_REQUEST,
            detail=extra_details or None
        )

class AuthorizationError(APIException):
    """授权错误异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 授权失败的具体描述
            **extra_details: 额外信息（如required_role、current_role等）
        """
        # 如果没有指定code，默认使用通用的UNAUTHORIZED
        code = extra_details.pop("code", "UNAUTHORIZED")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_401_UNAUTHORIZED,
            detail=extra_details or None
        )

class PrivilegeError(APIException):
    """权限错误异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 权限不足的具体描述
            **extra_details: 额外信息（如required_privilege、current_privilege等）
        """
        # 如果没有指定code，默认使用通用的FORBIDDEN
        code = extra_details.pop("code", "FORBIDDEN")
        
        super().__init__(
            code=code,
            message=message,
            http_status=status.HTTP_403_FORBIDDEN,
            detail=extra_details or None
        )

class InternalServerError(APIException):
    """服务器内部错误"""
    def __init__(self, message: str = "服务器内部错误", **extra_details):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=extra_details or None
        )


def handle_exception(e: Exception, msg: str) -> NoReturn:
    """异常管理标准化"""
    if isinstance(e, DataNotFoundException):
        raise NotFoundError(e.message)
    if isinstance(e, DataConflictException):
        raise ParamValueError(e.message)
    if isinstance(e, (DatabaseException)):
        logger.error(f"{msg}: {e.original_error}")
        raise InternalServerError(f"{msg}: {e.message}")
    if isinstance(e, (NotFoundError, ParamValueError, AuthorizationError, PrivilegeError, InternalServerError)):
        raise e
    logger.exception(f"{msg}: {e}")
    raise e