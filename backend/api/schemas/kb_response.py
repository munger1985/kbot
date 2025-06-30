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

class KBItem(BaseModel):
    """知识库项模型"""
    id: str = Field(..., description="知识库项ID")
    name: str = Field(..., description="知识库项名称")
    description: Optional[str] = Field(None, description="描述")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")

class KBUploadResponse(SuccessResponse):
    """知识库上传响应"""
    data: KBItem = Field(..., description="上传的知识库项")

class KBListResponse(SuccessResponse):
    """知识库列表响应"""
    data: List[KBItem] = Field(..., description="知识库项列表")

class KBPaginatedResponse(SuccessResponse):
    """知识库分页响应"""
    data: PaginatedResponse[KBItem] = Field(..., description="分页的知识库项")

class KBErrorResponse(ErrorResponse):
    """知识库错误响应"""
    error_type: str = Field("knowledge_base_error", description="错误类型")
    details: Optional[dict] = Field(None, description="错误详情")