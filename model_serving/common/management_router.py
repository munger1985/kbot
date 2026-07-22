"""Authenticated model-definition CRUD shared by the four model processes."""
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from .model_registry import ModelDefinitionNotFound, ModelRegistryService


class ModelCreateRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=256)
    model_name: str = Field(min_length=1, max_length=256)
    category: int = Field(gt=0)
    provider: str = Field(min_length=1, max_length=256)
    api_endpoint: str | None = Field(default=None, max_length=256)
    api_key: str | None = Field(default=None, max_length=4096)
    status: int = Field(default=0, ge=0, le=2)
    embedding_dimension: int | None = Field(default=None, gt=0)
    model_params: dict[str, Any] = Field(default_factory=dict)
    descs: str | None = Field(default=None, max_length=512)


class ModelUpdateRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    model_name: str | None = Field(default=None, min_length=1, max_length=256)
    provider: str | None = Field(default=None, min_length=1, max_length=256)
    api_endpoint: str | None = Field(default=None, max_length=256)
    api_key: str | None = Field(default=None, max_length=4096)
    status: int | None = Field(default=None, ge=0, le=2)
    embedding_dimension: int | None = Field(default=None, gt=0)
    model_params: dict[str, Any] | None = None
    descs: str | None = Field(default=None, max_length=512)


def create_model_management_router(*, category: int) -> APIRouter:
    router = APIRouter(prefix="/v1/models", tags=["Model Configuration"])

    def service(request: Request) -> ModelRegistryService:
        return request.app.state.model_registry

    @router.get("")
    async def list_models(request: Request):
        return {"models": await service(request).list(category=category)}

    @router.get("/{model_id}")
    async def get_model(model_id: int, request: Request):
        try:
            return await service(request).get(model_id, category=category)
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": str(exc)}) from exc

    @router.post("", status_code=status.HTTP_201_CREATED)
    async def create_model(payload: ModelCreateRequest, request: Request):
        if payload.category != category:
            raise HTTPException(status_code=422, detail={"code": "MODEL_CATEGORY_MISMATCH", "message": "model category does not belong to this process"})
        return await service(request).create(payload.model_dump(), actor_id=request.headers.get("X-KBot-Actor-Id", "svc:model-handler"))

    @router.patch("/{model_id}")
    async def update_model(model_id: int, payload: ModelUpdateRequest, request: Request):
        values = {key: value for key, value in payload.model_dump().items() if value is not None}
        try:
            return await service(request).update(model_id, values, actor_id=request.headers.get("X-KBot-Actor-Id", "svc:model-handler"))
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": str(exc)}) from exc

    @router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def archive_model(model_id: int, request: Request):
        try:
            await service(request).delete(model_id, actor_id=request.headers.get("X-KBot-Actor-Id", "svc:model-handler"))
        except Exception as exc:
            raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": str(exc)}) from exc

    return router
