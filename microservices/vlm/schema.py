from pydantic import BaseModel, Field
from typing import Any
import uuid
import time


# 定义请求模型
class VLMRequest(BaseModel):
    """VLM推理请求模型"""

    model_name: str = Field(..., description="模型名称")
    messages: list[dict[str, Any]] = Field(..., description="消息列表")
    max_tokens: int | None = Field(None, description="要生成的最大令牌数")
    temperature: float | None = Field(None, description="采样温度 (0.0-1.0，越低越确定)")
    stream: bool = Field(False, description="是否流式返回响应")
    timeout: int | None = Field(None, description="超时时间（秒）")
    top_p: float | None = Field(None, description="Top-p采样参数")
    frequency_penalty: float | None = Field(None, description="频率惩罚")
    presence_penalty: float | None = Field(None, description="存在惩罚")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_name: str = Field(..., description="模型名称")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")
    
# 定义响应模型
class VLMResponse(BaseModel):
    """VLM推理响应模型(兼容OpenAI)"""

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", description="响应流的唯一标识符")
    object: str = Field("chat.completion", description="对象类型，始终为 'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), description="响应创建时的Unix时间戳")
    model: str = Field(..., description="响应模型名称")
    choices: list[dict[str, Any]] = Field(..., description="包含响应消息的列表")
    usage: dict[str, int] = Field(..., description="令牌使用统计，包括 prompt_tokens、completion_tokens 和 total_tokens")
    processing_time: float = Field(..., description="处理时间（秒）（自定义字段）")
