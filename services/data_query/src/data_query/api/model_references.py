"""Model Serving 使用的 Data Query 引用检查接口。"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(
    prefix="/internal/v1/data-query/model-references",
    tags=["Data Query Model References"],
)


@router.get("/{model_id}")
async def list_model_references(
    model_id: UUID, request: Request,
) -> dict[str, object]:
    context = getattr(request.state, "auth_context", None)
    caller = None if context is None else (
        context.calling_service or context.client_id
    )
    if caller is None or not caller.startswith("kbot-model-"):
        raise HTTPException(status_code=403, detail="调用方无权读取模型引用")
    async with request.app.state.uow_factory() as uow:
        references = await uow.model_references.list_for_model(model_id=model_id)
    return {"model_id": str(model_id), "references": references}
