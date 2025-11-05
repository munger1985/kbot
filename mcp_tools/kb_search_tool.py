from loguru import logger
from typing import Any
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType, ToolType
from services.chat.agent_params import ToolParams
from services.search.kb_search_for_mcp import KBSearch
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository


class KBSearchTool(MCPTool):
    """知识库搜索工具"""
    
    def __init__(self, agent_id: int):
        super().__init__(
            tool_type=MCPToolType.KB_SEARCH,
            tool_name="knowledge_base_search",
            description="搜索知识库获取相关信息"
        )
        self.agent_id = agent_id
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "")
            # 如果 query 可能包含多个查询，可以按需分割
            if isinstance(query, str):
                # 单个查询字符串
                questions = [query]
            elif isinstance(query, list):
                # 已经是列表格式
                questions = query
            else:
                # 其他类型，转换为字符串并包装
                questions = [str(query)]

            search_type = parameters.get("search_type", "hybrid")
            limit = parameters.get("limit", 10)
            
            results = []
            metadata_list = []
            # 获取智能体的工具配置
            agent_conf_repo = KbotMdAgentConfRepository()
            kb_ids = await agent_conf_repo.get_unique_kb_id(self.agent_id)
            for kb_id in kb_ids:
                
        
                kb = KBSearch(agent_id=self.agent_id, kb_id=kb_id)
            
                result = await kb.search(
                        questions=questions,
                        search_type=search_type,
                        security=self.security,
                        tags=self.tags
                    )
            
                # 限制结果数量
                limited_result = result[:limit] if result else []
                metadata = {
                    "kb_id": kb_id,
                    "result_count": len(result),
                    "limit": limit
                }
            
                # 合并结果和元数据
                metadata_list.append(metadata)
                results.extend(limited_result)
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=results,
                confidence=0.9,
                metadata=metadata_list
            )
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[],
                confidence=0.0,
                metadata=[{"error": str(e)}]
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
                    "enum": ["vector", "fulltext", "summary", "hybrid"],
                    "description": "搜索类型：向量搜索、全文搜索、摘要搜索、混合搜索"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 10
                }
            },
            "required": ["query"]
        }