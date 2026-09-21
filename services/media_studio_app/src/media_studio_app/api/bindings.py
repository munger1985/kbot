"""多媒体创作工作台模型角色绑定内部 API。"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from media_studio_app.api.context import raise_application_error, require_actor
from media_studio_app.application import MediaStudioApplicationError, ModelBindingService, UpsertBindingCommand
from platform_core.contracts import AuthContext


router = APIRouter(prefix="/internal/v1/media-studio/bindings", tags=["MediaStudio Bindings"])


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BindingUpsertRequest(_Request):
    domain_id: int = Field(ge=1)
    role: Literal["IMAGE_GENERATION"] = "IMAGE_GENERATION"
    model_id: UUID
    expected_row_version: int | None = Field(default=None, ge=1)


def _service(request: Request) -> ModelBindingService:
    return request.app.state.binding_service


@router.get("")
async def list_bindings(domain_id: int, request: Request):
    require_actor(request, domain_id)
    return await _service(request).list(domain_id=domain_id)


@router.get("/model-references/{model_id}")
async def model_references(model_id: UUID, request: Request):
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext):
        raise_application_error(
            MediaStudioApplicationError(
                "AUTH_CONTEXT_REQUIRED", "缺少可信调用上下文", status_code=403,
            )
        )
    return {
        "references": await _service(request).model_references(model_id=model_id)
    }


@router.put("")
async def upsert_binding(payload: BindingUpsertRequest, request: Request):
    try:
        return await _service(request).upsert(UpsertBindingCommand(
            actor_id=require_actor(request, payload.domain_id),
            **payload.model_dump(),
        ))
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)
