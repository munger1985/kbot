from pydantic import BaseModel, Field
from typing import Generic, TypeVar

T = TypeVar('T')

class SuccessResponse(BaseModel, Generic[T]):
    """KBOT API 成功响应模型"""
    message: str = Field("Success", description="返回的响应信息，用于前端显示给用户")
    data: T | None = Field(default=None, description="响应返回的业务数据")
    
    # 显式设置模型配置
    model_config = {
        "arbitrary_types_allowed": True,
    }