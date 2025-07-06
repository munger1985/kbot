from typing import Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')

class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(..., description="请求是否成功")
    message: str = Field(..., description="响应消息")
    code: int = Field(200, description="状态码")

class SuccessResponse(BaseResponse):
    """成功响应模型"""
    success: bool = Field(True, description="请求成功")
    message: str = Field("操作成功", description="成功消息")

class SuccessQueryResponse(SuccessResponse):
    """成功查询响应模型"""
    data: Optional[List[dict]] = Field(None, description="返回结果集")

class SuccessWithErrorResponse(SuccessResponse):
    """部分失败的成功响应模型"""
    code: int = Field(207, description="多状态码")
    details: Optional[dict] = Field(None, description="详情")

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    success: bool = Field(False, description="请求失败")
    code: int = Field(400, description="错误码")
    error_type: str = Field(..., description="错误类型")
    details: Optional[dict] = Field(None, description="错误详情")

class Pagination(BaseModel):
    """分页信息"""
    total: int = Field(..., description="总记录数")
    page: int = Field(1, description="当前页码")
    page_size: int = Field(10, description="每页数量")

class PaginatedResponse(GenericModel, Generic[T]):
    """分页响应模型"""
    items: List[T] = Field(..., description="数据列表")
    pagination: Pagination = Field(..., description="分页信息")
