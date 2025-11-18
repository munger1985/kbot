from abc import ABC, abstractmethod
from typing import Any
from loguru import logger
from core.dictionary import MCPToolType
from pydantic import BaseModel, Field

 
class Tool(BaseModel):
    """工具定义"""
    tool_type: MCPToolType = Field(..., description="工具类型")
    tool_name: str = Field(..., description="工具函数名称")
    description: str = Field(..., description="工具函数描述")
    parameters: dict[str, Any] = Field(..., description="工具函数参数schema")

class KBSearchResult(BaseModel):
    """知识库搜索结果"""
    file_id: str = ""
    chunk_type: int = 1
    page_num: int = 0
    chunk_file_path: str = ""
    content: str = ""
    similarity: float = 0.0
    weight: float = 0.0
    reranker_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True

class InternetSearchResult(BaseModel):
    """网络搜索结果"""
    title: str = ""
    url: str = ""
    content: str = ""
    snippet: str = ""
    relevance_score: float = 0.0
    weight: float = 0.0
    reranker_score: float = 0.0

class CalculatorResult(BaseModel):
    """计算器结果"""
    expression: str = ""
    result: str = ""
    steps: list[str] = []  # 计算步骤
    confidence: float = 1.0

class ToolResult(BaseModel):
    """最终组合结果 - 三种结果的并集"""
    tool_type: MCPToolType
    
    # 明确表示是三种结果之一的并集
    kb_results: list[KBSearchResult] = []
    internet_results: list[InternetSearchResult] = []
    calculator_result: CalculatorResult | None = None
    
    confidence: float = 0.0
    metadata: list[dict[str, Any]] = []
    
    

class MCPTool(ABC):
    """MCP工具基类"""
    
    def __init__(self, tool_type: MCPToolType, tool_name: str, description: str):
        self.tool_type = tool_type
        self.tool_name = tool_name
        self.description = description
        self.security = 0
        self.tags = []
    
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
            logger.debug(f"注销MCP工具: {tool_name}")
    
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