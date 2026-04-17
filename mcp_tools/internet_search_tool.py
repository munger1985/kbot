from loguru import logger
from typing import Any
import aiohttp
import json
import asyncio
from mcp_tools import MCPTool
from core.dictionary import MCPToolType
from .base import ToolResult


class InternetSearchTool(MCPTool):
    """通用网络搜索工具 - 适配新ToolResult结构"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.INTERNET_SEARCH,
            tool_name="internet_search",
            description="搜索互联网获取最新信息"
        )
        self.session: aiohttp.ClientSession | None = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建aiohttp会话"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=15, connect=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                }
            )
        return self.session
    
    
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """执行搜索 - 适配新ToolResult结构"""
        return ToolResult(
            tool_type=self.tool_type,
            internet_results=[],
            confidence=0.1,
            metadata=[{
                "error": "网络搜索服务当前不可用"
            }]
        )
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
    
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