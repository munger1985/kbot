from fastapi import APIRouter

from .controllers.health import router as health_router
from .routers.kb_upload_router import router as kb_upload_router

router = APIRouter()

router.include_router(health_router, tags=["health"])
router.include_router(kb_upload_router, tags=["knowledge-base"])