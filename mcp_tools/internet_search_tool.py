from loguru import logger
from typing import Any, List, Dict
import aiohttp
from mcp_tools import MCPTool, ToolResult
from core.dictionary import MCPToolType


class InternetSearchTool(MCPTool):
    """网络搜索工具 - 使用DuckDuckGo搜索API"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.INTERNET_SEARCH,
            tool_name="internet_search",
            description="搜索互联网获取最新信息"
        )
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建aiohttp会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _search_duckduckgo(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """使用DuckDuckGo API进行搜索"""
        session = await self._get_session()
        
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_duckduckgo_results(data, query, limit)
                else:
                    logger.warning(f"DuckDuckGo API返回状态码: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"DuckDuckGo搜索失败: {e}")
            return []
    
    def _parse_duckduckgo_results(self, data: Dict, query: str, limit: int) -> List[Dict[str, Any]]:
        """解析DuckDuckGo返回结果"""
        results = []
        
        # 处理直接答案
        if data.get('AbstractText'):
            results.append({
                "content": f"直接答案: {data['AbstractText']}",
                "weight": 0.9,
                "reranker_score": 0.9,
                "source": data.get('AbstractSource', 'DuckDuckGo'),
                "url": data.get('AbstractURL', '')
            })
        
        # 处理相关主题
        for topic in data.get('RelatedTopics', [])[:limit]:
            if 'Text' in topic and 'FirstURL' in topic:
                results.append({
                    "content": topic['Text'],
                    "weight": 0.7,
                    "reranker_score": 0.7,
                    "source": "DuckDuckGo Related Topics",
                    "url": topic['FirstURL']
                })
        
        # 如果结果不足，添加一些基础信息
        if len(results) == 0:
            results.append({
                "content": f"关于'{query}'的搜索结果，请参考相关网络资源获取最新信息。",
                "weight": 0.5,
                "reranker_score": 0.5,
                "source": "Internet Search",
                "url": ""
            })
        
        return results[:limit]
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        try:
            query = parameters.get("query", "").strip()
            limit = min(parameters.get("limit", 5), 10)  # 限制最大数量
            
            if not query:
                raise ValueError("搜索查询不能为空")
            
            logger.info(f"执行网络搜索: {query}, 限制: {limit}")
            
            # 执行搜索
            search_results = await self._search_duckduckgo(query, limit)
            
            if not search_results:
                return ToolResult(
                    tool_type=self.tool_type,
                    tool_name=self.tool_name,
                    content=[],
                    confidence=0.3,
                    metadata=[{
                        "search_query": query, 
                        "result_count": 0,
                        "message": "未找到相关搜索结果"
                    }]
                )
            
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=search_results,
                confidence=0.8,
                metadata=[{
                    "search_query": query, 
                    "result_count": len(search_results),
                    "search_engine": "DuckDuckGo"
                }]
            )
            
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return ToolResult(
                tool_type=self.tool_type,
                tool_name=self.tool_name,
                content=[],
                confidence=0.0,
                metadata=[{"error": str(e)}]
            )
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "搜索查询语句"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }