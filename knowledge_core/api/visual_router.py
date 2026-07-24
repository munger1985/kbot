"""Knowledge Core 多图片视觉检索内部端点。"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from platform_core.contracts import INTERNAL_API_V1
from platform_core.security import require_domain_match


router = APIRouter(
    prefix=f"{INTERNAL_API_V1}/knowledge/retrieval",
    tags=["Knowledge Core Visual Retrieval"],
)


class VisualSearchRequest(BaseModel):
    domain_id: int = Field(gt=0)
    agent_id: UUID
    collection_ids: list[UUID] = Field(min_length=1, max_length=32)
    images_base64: list[str] = Field(min_length=1, max_length=8)
    per_image_limit: int = Field(default=10, ge=1, le=50)
    result_limit: int = Field(default=20, ge=1, le=100)

    @field_validator("images_base64")
    @classmethod
    def validate_images(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("查询图片不能为空")
        if sum(len(value) for value in values) > 32 * 1024 * 1024:
            raise ValueError("查询图片总大小超过限制")
        return values


@router.post("/visual")
async def search_visual(
    payload: VisualSearchRequest, request: Request
) -> dict:
    require_domain_match(request, payload.domain_id)
    try:
        scoped = (
            await request.app.state.kc_scope_service.resolve_agent_collections(
                domain_id=payload.domain_id,
                agent_id=payload.agent_id,
                collection_ids=payload.collection_ids,
            )
        )
        if set(scoped) != set(payload.collection_ids):
            raise ValueError("一个或多个 Collection 不可用")
        outcome = await request.app.state.kc_visual_service.search(
            collection_ids=list(scoped),
            images_base64=payload.images_base64,
            per_image_limit=payload.per_image_limit,
            result_limit=payload.result_limit,
        )
        return outcome
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_VISUAL_QUERY", "message": str(exc)},
        ) from exc
