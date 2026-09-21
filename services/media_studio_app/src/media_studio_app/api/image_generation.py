"""多媒体创作工作台文生图内部 API。"""

from uuid import UUID

from fastapi import APIRouter, Header, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field

from media_studio_app.api.context import (
    auth_context, idempotency_key, raise_application_error, require_actor, run_response,
)
from media_studio_app.application import (
    MediaStudioApplicationError, CreateImageGenerationCommand, ImageGenerationService,
)


router = APIRouter(
    prefix="/internal/v1/media-studio/image-generations/runs",
    tags=["MediaStudio Image Generation"],
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageGenerationCreateRequest(_Request):
    domain_id: int = Field(ge=1)
    prompt: str = Field(min_length=1, max_length=32000)
    aspect_ratio: str = Field(default="1:1", pattern=r"^[1-9][0-9]?:[1-9][0-9]?$")
    count: int = Field(default=1, ge=1, le=4)


def _service(request: Request) -> ImageGenerationService:
    return request.app.state.image_generation_service


@router.get("")
async def list_image_runs(
    domain_id: int,
    request: Request,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    actor_id = require_actor(request, domain_id)
    return await _service(request).list(
        domain_id=domain_id, actor_id=actor_id, status=status, limit=limit,
    )


@router.post("")
async def create_image_run(
    payload: ImageGenerationCreateRequest,
    request: Request,
    idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
):
    context = auth_context(request)
    try:
        view, created = await _service(request).create(CreateImageGenerationCommand(
            actor_id=require_actor(request, payload.domain_id),
            request_id=context.request_id,
            trace_id=context.trace_id,
            idempotency_key=idempotency_key(request, idempotency_key_header),
            **payload.model_dump(),
        ))
        return run_response(view, created)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)


@router.get("/{run_id}")
async def get_image_run(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        return await _service(request).get(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_run(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        await _service(request).delete(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)


@router.get("/{run_id}/events")
async def list_image_events(run_id: UUID, domain_id: int, request: Request):
    actor_id = require_actor(request, domain_id)
    try:
        return await _service(request).list_events(domain_id=domain_id, run_id=run_id, actor_id=actor_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)
