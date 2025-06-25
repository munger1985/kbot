from fastapi import APIRouter

from .controllers.health import router as health_router

router = APIRouter()

router.include_router(health_router, prefix="/api", tags=["health"])