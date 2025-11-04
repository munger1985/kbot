from loguru import logger
from typing import Any
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType
from services.chat.agent_params import ToolParams
from services.search.kb_search import KBSearch


class KBSearchTool(MCPTool):
    """知识库搜索工具"""
    
    def __init__(self, tool_params: ToolParams):
        super().__init__(
            tool_type=MCPToolType.KB_SEARCH,
            tool_name="knowledge_base_search",
            description="搜索知识库获取相关信息"
        )
        self.tool_params = tool_params
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
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
            
            # 限制结果数量
            limited_result = result[:limit] if result else []
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=limited_result,
                confidence=0.9,
                metadata={
                    "search_type": search_type, 
                    "result_count": len(limited_result),
                    "total_result_count": len(result) if result else 0
                }
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
    
    def get_schema(self) -> dict[str, Any]:
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
                    "description": "搜索类型：向量搜索、全文搜索、混合搜索"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 10
                }
            },
            "required": ["query"]
        }