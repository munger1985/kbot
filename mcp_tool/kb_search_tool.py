from loguru import logger
from typing import Dict, Any
from mcp_tool import MCPTool, ToolType, ToolResult
from services.chat.agent_params import ToolParams
from services.search.kb_search import KBSearch


class KBSearchTool(MCPTool):
    """知识库搜索工具"""
    
    def __init__(self, tool_params: ToolParams):
        super().__init__(
            tool_type=ToolType.KB_SEARCH,
            tool_name="knowledge_base_search",
            description="搜索知识库获取相关信息"
        )
        self.tool_params = tool_params
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "")
            search_type = parameters.get("search_type", "hybrid")
            limit = parameters.get("limit", 10)
            
            kb = KBSearch(self.tool_params)
            
            if search_type == "vector":
                result = await kb.search(
                    vector_search_question=query,
                    full_text_question=[],
                    security=self.security,
                    tags=self.tags
                )
            elif search_type == "fulltext":
                result = await kb.search(
                    vector_search_question="",
                    full_text_question=[query],
                    security=self.security,
                    tags=self.tags
                )
            else:  # hybrid
                result = await kb.search(
                    vector_search_question=query,
                    full_text_question=[query],
                    security=self.security,
                    tags=self.tags
                )
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=result[:limit] if result else [],
                confidence=0.9,
                metadata={"search_type": search_type, "result_count": len(result) if result else 0}
            )
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询语句"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["vector", "fulltext", "hybrid"],
                    "description": "搜索类型"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制"
                }
            },
            "required": ["query"]
        }

class InternetSearchTool(MCPTool):
    """网络搜索工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=ToolType.INTERNET_SEARCH,
            tool_name="internet_search",
            description="搜索互联网获取最新信息"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "")
            # 这里实现网络搜索逻辑
            # 例如调用搜索引擎API
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=f"网络搜索结果: {query}",
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content="网络搜索失败",
                confidence=0.0
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索查询"}
            },
            "required": ["query"]
        }

class CalculatorTool(MCPTool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=ToolType.CALCULATOR,
            tool_name="calculator",
            description="执行数学计算"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        try:
            expression = parameters.get("expression", "")
            # 这里实现计算逻辑
            result = eval(expression)  # 注意：生产环境需要更安全的计算方式
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=str(result),
                confidence=1.0
            )
        except Exception as e:
            logger.error(f"计算失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content="计算失败",
                confidence=0.0
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式"}
            },
            "required": ["expression"]
        }