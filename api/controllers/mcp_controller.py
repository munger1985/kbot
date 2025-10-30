from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import json
from loguru import logger

from api.schemas.mcp_schema import KBSearchRequest, KBSearchResponse, StreamChunk, KBResultResponse
from services.mcp.kb_search import mcp_search_service


class MCPKBSearchController:
    """MCP知识库搜索控制器"""
    
    def __init__(self):
        self.router = APIRouter(prefix="/mcp/kb-search", tags=["MCP Knowledge Base Search"])
        self._register_routes()
    
    def _register_routes(self):
        """注册路由"""
        self.router.add_api_route(
            "/search",
            self.search_knowledge_base,
            methods=["POST"],
            response_model=KBSearchResponse,
            summary="知识库搜索"
        )
        
        self.router.add_api_route(
            "/search-stream",
            self.stream_search_knowledge_base,
            methods=["POST"],
            summary="流式知识库搜索"
        )
    
    async def search_knowledge_base(self, request: KBSearchRequest) -> KBSearchResponse:
        """
        知识库搜索接口
        
        Args:
            request: 搜索请求参数
            
        Returns:
            KBSearchResponse: 搜索结果
        """
        try:
            # 收集流式结果
            results = []
            async for chunk in mcp_search_service.stream_kb_search(request):
                if chunk.type == "result" and chunk.data:
                    results.append(chunk.data)
                elif chunk.type == "error":
                    raise HTTPException(status_code=500, detail=chunk.message)
            
            # 转换结果
            kb_results = []
            for result_data in results:
                kb_results.append(KBResultResponse(**result_data))
            
            return KBSearchResponse(
                success=True,
                results=kb_results,
                total_count=len(kb_results),
                message=f"搜索完成，找到 {len(kb_results)} 条结果"
            )
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
    
    async def stream_search_knowledge_base(self, request: KBSearchRequest):
        """
        流式知识库搜索接口
        
        Args:
            request: 搜索请求参数
            
        Returns:
            StreamingResponse: 流式响应
        """
        async def generate():
            """生成流式响应"""
            async for chunk in mcp_search_service.stream_kb_search(request):
                yield f"data: {json.dumps(chunk.dict(), ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )


# 创建控制器实例
mcp_kb_search_controller = MCPKBSearchController()