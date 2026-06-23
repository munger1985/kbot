from fastapi import APIRouter
from .health import router as health_router
from .kb_router import router as kb_router
from .agent_router import router as agent_router
# from .log_router import router as log_router
from .auth_router import router as auth_router
from .ops_router import router as ops_router
from .parser_router import router as parser_router

router = APIRouter(prefix="/api")

# 合并路由
router.include_router(agent_router)
router.include_router(health_router)
router.include_router(kb_router)
# router.include_router(log_router)
router.include_router(auth_router)
router.include_router(ops_router)
router.include_router(parser_router)

__all__ = ["router"]