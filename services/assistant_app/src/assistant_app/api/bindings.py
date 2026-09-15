"""智能工作台模型角色绑定内部 API。"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from assistant_app.api.context import raise_application_error, require_actor
from assistant_app.application import AssistantApplicationError, ModelBindingService, UpsertBindingCommand


router = APIRouter(prefix="/internal/v1/assistant/bindings", tags=["Assistant Bindings"])


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BindingUpsertRequest(_Request):
    domain_id: int = Field(ge=1)
    role: Literal["KNOWLEDGE", "X_SEARCH", "IMAGE_GENERATION"]
    model_id: UUID
    expected_row_version: int | None = Field(default=None, ge=1)


def _service(request: Request) -> ModelBindingService:
    return request.app.state.binding_service


@router.get("")
async def list_bindings(domain_id: int, request: Request):
    require_actor(request, domain_id)
    return await _service(request).list(domain_id=domain_id)


@router.put("")
async def upsert_binding(payload: BindingUpsertRequest, request: Request):
    try:
        return await _service(request).upsert(UpsertBindingCommand(
            actor_id=require_actor(request, payload.domain_id),
            **payload.model_dump(),
        ))
    except AssistantApplicationError as exc:
        raise_application_error(exc)
