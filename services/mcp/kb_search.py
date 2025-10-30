import asyncio
from loguru import logger
from typing import AsyncGenerator, List, Optional
from api.schemas.mcp_schema import KBSearchRequest, KBResultResponse, StreamChunk
from chat.agent_params import ToolParams, KBResult
from core.dictionary import KBSearchType as InternalKBSearchType
from .kb_search import KBSearch


class MCPSearchService:
    """MCP知识库搜索服务"""
    
    def __init__(self):
        self.active_searches: dict[str, asyncio.Task] = {}
    
    async def stream_kb_search(self, request: KBSearchRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        流式知识库搜索
        
        Args:
            request: 搜索请求参数
            
        Yields:
            StreamChunk: 流式响应数据块
        """
        search_id = f"search_{id(request)}"
        
        try:
            # 转换搜索类型
            search_type_mapping = {
                "vector": InternalKBSearchType.VECTOR.value,
                "fulltext": InternalKBSearchType.FULLTEXT.value,
                "summary": InternalKBSearchType.SUMMARY.value,
                "graph": InternalKBSearchType.GRAPH.value
            }
            
            internal_search_type = search_type_mapping.get(request.search_type.value)
            if not internal_search_type:
                yield StreamChunk(
                    type="error",
                    message=f"不支持的搜索类型: {request.search_type}"
                )
                return
            
            # 创建工具参数
            tool_params = ToolParams()
            tool_params.tool_id = request.tool_id
            tool_params.search_type = internal_search_type
            tool_params.threshold = request.threshold
            tool_params.top_k = request.top_k
            tool_params.tool_weight = request.tool_weight
            
            # 创建搜索实例
            kb_search = KBSearch(tool_params)
            
            # 执行搜索
            logger.info(f"开始MCP知识库搜索: {request.tool_id}, 类型: {request.search_type}")
            
            results: Optional[List[KBResult]] = await kb_search.search(
                vector_search_question=request.vector_search_question,
                full_text_question=request.full_text_question,
                security=request.security,
                tags=request.tags
            )
            
            if not results:
                yield StreamChunk(
                    type="complete",
                    message="搜索完成，未找到相关结果",
                    total=0,
                    current=0
                )
                return
            
            # 转换结果并流式返回
            total = len(results)
            for i, result in enumerate(results):
                response_result = KBResultResponse(
                    file_id=result.file_id,
                    chunk_type=result.chunk_type,
                    page_num=result.page_num,
                    content=result.content,
                    similarity=result.similarity,
                    weight=result.weight
                )
                
                yield StreamChunk(
                    type="result",
                    data=response_result.dict(),
                    total=total,
                    current=i + 1
                )
                
                # 模拟处理延迟，实际使用时可以根据需要调整
                await asyncio.sleep(0.01)
            
            # 发送完成信号
            yield StreamChunk(
                type="complete",
                message="搜索完成",
                total=total,
                current=total
            )
            
            logger.info(f"MCP知识库搜索完成: 找到 {total} 条结果")
            
        except Exception as e:
            logger.error(f"MCP知识库搜索错误: {str(e)}")
            yield StreamChunk(
                type="error",
                message=f"搜索过程中发生错误: {str(e)}"
            )
    
    async def cancel_search(self, search_id: str):
        """取消搜索任务"""
        if search_id in self.active_searches:
            self.active_searches[search_id].cancel()
            del self.active_searches[search_id]
            logger.info(f"已取消搜索任务: {search_id}")


# 全局服务实例
mcp_search_service = MCPSearchService()