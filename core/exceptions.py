# exceptions.py
from fastapi import HTTPException, status
from typing import Any

# --------------------------------------------------
# 1. 基础设施层异常（DAO层使用）
# --------------------------------------------------
class DataAccessException(Exception):
    """数据访问异常基类"""
    pass

class DatabaseException(DataAccessException):
    """数据库异常"""
    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error

class NotFoundException(DataAccessException):
    """记录不存在异常"""
    def __init__(self, resource: str, identifier: Any):
        self.resource = resource
        self.identifier = identifier

class DuplicateRecordException(DataAccessException):
    """重复记录异常"""
    def __init__(self, resource: str, field: str, value: Any):
        self.resource = resource
        self.field = field
        self.value = value

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

class ResourceNotFoundException(APIException):
    """资源不存在异常 - 极简版"""
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

class UnauthorizedException(APIException):
    """未认证异常"""
    def __init__(self, message: str = "请先登录", **extra_details):
        super().__init__(
            code="UNAUTHORIZED",
            message=message,
            http_status=status.HTTP_401_UNAUTHORIZED,
            detail=extra_details or None
        )

class ForbiddenException(APIException):
    """无权限异常"""
    def __init__(self, message: str = "权限不足", **extra_details):
        super().__init__(
            code="FORBIDDEN",
            message=message,
            http_status=status.HTTP_403_FORBIDDEN,
            detail=extra_details or None
        )

class BadRequestException(APIException):
    """请求错误异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 消息模板，如"余额不足，需要{required}，当前{current}"
            **kwargs: 格式化参数
        """
        super().__init__(
            code="BAD_REQUEST",
            message=message,
            http_status=status.HTTP_400_BAD_REQUEST,
            detail=extra_details or None
        )

class ValidationException(APIException):
    """参数验证异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 直接显示给用户的错误信息
            **extra_details: 额外的调试信息（可选）
        """
        super().__init__(
            code="VALIDATION_ERROR",  # 固定错误码
            message=message,          # 直接使用传入的消息
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=extra_details or None  # 可选附加信息
        )

class ConflictException(APIException):
    """资源冲突异常"""
    def __init__(self, message: str, **extra_details):
        """
        Args:
            message: 冲突的具体描述
            **extra_details: 额外信息（如resource、field、value等）
        """
        super().__init__(
            code="CONFLICT",
            message=message,
            http_status=status.HTTP_409_CONFLICT,
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