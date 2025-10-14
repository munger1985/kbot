from fastapi import APIRouter
from .health import router as health_router
from .kb_router import router as kb_router
from .agent_router import router as agent_router
from .security_router import router as security_router
from .model_router import router as model_router
from .log_router import router as log_router

router = APIRouter(prefix="/api")

# 合并路由
router.include_router(agent_router)
router.include_router(health_router)
router.include_router(kb_router)
router.include_router(security_router)
router.include_router(model_router)
router.include_router(log_router)

__all__ = ["router"]