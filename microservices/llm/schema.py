
from pydantic import BaseModel, Field
from typing import Any
import uuid
import time

# ==================== MCP工具相关模型定义 ====================

class ToolFunction(BaseModel):
    """工具函数定义"""
    name: str = Field(..., description="工具函数名称")
    description: str = Field(..., description="工具函数描述")
    parameters: dict[str, Any] = Field(..., description="工具函数参数schema")

class Tool(BaseModel):
    """工具定义"""
    type: str = Field("function", description="工具类型")
    function: ToolFunction = Field(..., description="工具函数定义")

class ToolCall(BaseModel):
    """工具调用请求"""
    id: str = Field(..., description="工具调用ID")
    type: str = Field("function", description="工具调用类型")
    function: dict[str, Any] = Field(..., description="函数调用详情")

# ==================== 聊天请求/响应模型 ====================

class ChatResponse(BaseModel):
    """聊天响应模型(兼容OpenAI)"""

    id: str = Field(default_factory=lambda: f"sse-{uuid.uuid4()}", description="响应流的唯一标识符")
    object: str = Field("chat.completion", description="对象类型，始终为'chat.completion'")
    created: int = Field(default_factory=lambda: int(time.time()), description="响应创建时的Unix时间戳")
    model: str = Field(..., description="响应模型名称")
    choices: list[dict[str, Any]] = Field(..., description="包含消息的聊天完成选项列表")
    usage: dict[str, int] = Field(..., description="令牌使用统计，包括prompt_tokens、completion_tokens和total_tokens")
    processing_time: float = Field(..., description="处理时间（秒）（自定义字段）")
    tool_calls: list[ToolCall] | None = Field(None, description="工具调用列表")


class ChatRequest(BaseModel):
    """聊天请求模型"""

    model_id: int = Field(..., description="要使用的特定模型ID")
    messages: list[dict[str, str]] | str = Field(..., description="聊天消息列表")
    max_tokens: int | None = Field(None, description="要生成的最大令牌数")
    temperature: float | None = Field(None, description="采样温度（0.0-1.0，越低越确定）")
    stream: bool = Field(False, description="是否流式传输响应")
    timeout: int | None = Field(None, description="超时时间（秒）")
    top_p: float | None = Field(None, description="Top-p采样参数")
    frequency_penalty: float | None = Field(None, description="频率惩罚")
    presence_penalty: float | None = Field(None, description="存在惩罚")
    tools: list[dict[str, Any]] | None = Field(None, description="工具列表，支持MCP工具调用")
    tool_choice: str | None = Field(None, description="工具选择策略")
    enable_tool_calls: bool = Field(False, description="是否启用工具调用")

class ToggleModelRequest(BaseModel):
    """启用或禁用模型请求表单。"""
    model_id: int = Field(..., description="模型唯一标识符")
    operation: str = Field(..., description="操作类型，'load' 或 'unload'")
