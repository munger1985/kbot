from fastapi import APIRouter
from api.controllers.mcp_controller import mcp_kb_search_controller

# 创建路由
router = APIRouter()

# 包含控制器路由
router.include_router(mcp_kb_search_controller.router)

@router.get("/mcp/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "MCP Knowledge Base Search"}

@router.get("/v1")
async def root():
    return {"message": "MCP Knowledge Base Search Service"}