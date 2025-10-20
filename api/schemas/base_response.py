from pydantic import BaseModel, Field
from fastapi import status

class SuccessResponse(BaseModel):
    code: int = Field(status.HTTP_200_OK, description="响应状态码")
    message: str = Field("Success", description="返回的响应信息")
    success: bool = Field(True, description="请求响应状态")

class ErrorResponse(BaseModel):
    code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
    message: str = Field("Error", description="返回的响应信息")
    success: bool = Field(False, description="请求响应状态")

class SuccessQueryResponse(SuccessResponse):
    """查询成功响应模型"""
    data: dict | list[dict] = Field(..., description="响应返回的数据")

class SuccessWithErrorResponse(SuccessResponse):
    """部分失败的成功响应模型"""
    code: int = Field(status.HTTP_207_MULTI_STATUS, description="多状态码")
    details: dict | None = Field(None, description="详情")