from fastapi import APIRouter

router = APIRouter(tags=["Health Check"])

@router.get("/health", summary="健康检查")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}