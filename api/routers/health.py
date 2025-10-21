from fastapi import APIRouter

router = APIRouter(tags=["Health Check"])

@router.get("/health", summary="健康检查")
async def health_check():
    """健康检查接口
    
    Returns:
    - **dict**: 包含服务状态的响应数据
    ```
        {
        "status": "ok"
        }
    ```
    """
    return {"status": "ok"}