"""Model Serving 对外暴露的 OpenAI 兼容模型目录。"""

from fastapi import APIRouter, Request

from platform_core.contracts import PUBLIC_API_V1


def create_openai_models_router(*, category: int) -> APIRouter:
    """只公开当前进程能够推理的已启用模型。"""
    router = APIRouter(prefix=PUBLIC_API_V1, tags=["OpenAI Compatible"])

    @router.get("/models")
    async def list_models(request: Request) -> dict:
        rows = await request.app.state.model_registry.list(category=category)
        data = [
            {
                "id": row["served_model_name"],
                "object": "model",
                "created": 0,
                "owned_by": "kbot-model-serving",
            }
            for row in rows
            if int(row.get("status") or 0) == 1
        ]
        return {"object": "list", "data": data}

    return router
