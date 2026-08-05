"""供模型目录执行归档和删除引用检查。"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import get_auth_context


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/model-references",
    tags=["Knowledge Core Model References"],
)


@router.get("/{model_id}")
async def list_model_references(model_id: UUID, request: Request) -> dict:
    context = get_auth_context(request)
    caller = context.calling_service or context.client_id
    if caller != "kbot-main-api" and not caller.startswith("kbot-model-"):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "MODEL_REFERENCE_AUDIT_DENIED",
                "message": "调用方无权检查模型引用",
            },
        )
    rows = await request.app.state.kc_model_reference_service.list(
        model_id=model_id
    )
    return {
        "references": [
            {
                "service": "knowledge-core",
                "resource_type": str(row["resource_type"]),
                "resource_id": str(row["resource_id"]),
                "usage": str(row["binding_role"]),
            }
            for row in rows
        ]
    }
