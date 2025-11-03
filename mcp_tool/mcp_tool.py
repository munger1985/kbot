from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
from dataclasses import dataclass
from loguru import logger

class ToolType(Enum):
    """工具类型枚举"""
    KB_SEARCH = "kb_search"
    FUNCTION_CALL = "function_call"
    INTERNET_SEARCH = "internet_search"
    AGENT_CALL = "agent_call"
    CHAT_AI = "chat_ai"
    CALCULATOR = "calculator"
    CODE_EXECUTION = "code_execution"

@dataclass
class ToolCall:
    """工具调用请求"""
    tool_type: ToolType
    tool_name: str
    parameters: dict[str, Any]
    description: str = ""

@dataclass
class ToolResult:
    """工具执行结果"""
    tool_type: ToolType
    tool_name: str
    content: Any
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None

class MCPTool(ABC):
    """MCP工具基类"""
    
    def __init__(self, tool_type: ToolType, tool_name: str, description: str):
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.description = description
    
    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """获取工具的参数schema"""
        pass

class MCPToolRegistry:
    """MCP工具注册表"""
    
    def __init__(self):
        self._tools: dict[str, MCPTool] = {}
    
    def register(self, tool: MCPTool):
        """注册工具"""
        self._tools[tool.tool_name] = tool
        logger.debug(f"注册MCP工具: {tool.tool_name}")
    
    def unregister(self, tool_name: str):
        """注销工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
    
    def get_tool(self, tool_name: str) -> MCPTool | None:
        """获取工具"""
        return self._tools.get(tool_name)
    
    def get_all_tools(self) -> list[MCPTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tools_schema(self) -> dict[str, Any]:
        """获取所有工具的schema"""
        return {
            tool.tool_name: {
                "type": tool.tool_type.value,
                "description": tool.description,
                "schema": tool.get_schema()
            }
            for tool in self._tools.values()
        }