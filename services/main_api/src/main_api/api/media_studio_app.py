"""多媒体创作工作台公开 BFF 路由。"""

from typing import Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from main_api.api.models import ModelCatalogItem, load_model_catalog
from main_api.application import UserAuthService, authorize_app_request, authorize_app_request_any
from platform_clients import MediaStudioClient
from platform_core.contracts import PUBLIC_API_V1


router = APIRouter(prefix=f"{PUBLIC_API_V1}/apps/media-studio", tags=["Media Studio App"])
TERMINAL_STATUSES = frozenset({"COMPLETED", "FAILED", "REJECTED"})


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MediaStudioLoginPayload(_Payload):
    user_id: str = Field(min_length=1, max_length=256)
    password: str = Field(min_length=1, max_length=256)


class MediaStudioPasswordChangePayload(_Payload):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        if not (
            any(char.islower() for char in value)
            and any(char.isupper() for char in value)
            and any(char.isdigit() for char in value)
            and any(not char.isalnum() for char in value)
        ):
            raise ValueError("新密码必须同时包含大小写字母、数字和特殊字符")
        return value


class MediaStudioBindingUpsertPayload(_Payload):
    role: Literal["IMAGE_GENERATION"] = "IMAGE_GENERATION"
    model_id: UUID
    expected_row_version: int | None = Field(default=None, ge=1)


class MediaStudioImageCreatePayload(_Payload):
    prompt: str = Field(min_length=1, max_length=32000)
    aspect_ratio: str = Field(default="1:1", pattern=r"^[1-9][0-9]?:[1-9][0-9]?$")
    count: int = Field(default=1, ge=1, le=4)


def _client(request: Request) -> MediaStudioClient:
    return cast(MediaStudioClient, request.app.state.media_studio_client)


async def _require(request: Request, permission: str):
    return await authorize_app_request(request, app_id="media_studio", permission=permission)


def _run_response(view: dict[str, Any]) -> JSONResponse:
    code = 200 if view.get("status") in TERMINAL_STATUSES else 202
    return JSONResponse(status_code=code, content=view)


@router.post("/auth/login")
async def login(payload: MediaStudioLoginPayload, request: Request):
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.login(app_id="media_studio", user_id=payload.user_id, password=payload.password)


@router.post("/auth/password")
async def change_password(payload: MediaStudioPasswordChangePayload, request: Request):
    service = cast(UserAuthService, request.app.state.user_auth_service)
    return await service.change_password(
        claims=request.state.user_token_claims,
        current_password=payload.current_password,
        new_password=payload.new_password,
    )


@router.get("/access")
async def get_access(request: Request):
    domain_id, _, snapshot = await _require(request, "media_studio:use")
    bindings = await _client(request).list_bindings(
        domain_id=domain_id, auth_context=request.state.auth_context,
    )
    image = next((item for item in bindings if item.get("role") == "IMAGE_GENERATION"), None)
    return {
        "app_id": snapshot.app_id,
        "domain_id": snapshot.domain_id,
        "user_id": snapshot.user_id,
        "roles": snapshot.roles,
        "permissions": sorted(snapshot.permissions),
        "bindings": bindings,
        "capabilities": {
            "image_generation": {
                "bound": image is not None,
                "verified": bool(image and image.get("supports_image_generation")),
                "ready": bool(image and image.get("ready")),
            }
        },
    }


@router.get("/model-catalog", response_model=list[ModelCatalogItem])
async def list_model_catalog(request: Request):
    await authorize_app_request_any(
        request,
        app_id="media_studio",
        permissions=("media_studio:model_binding_manage", "media_studio:image_generate"),
    )
    return await load_model_catalog(request)


@router.get("/bindings")
async def list_bindings(request: Request):
    domain_id, _, _ = await _require(request, "media_studio:model_binding_manage")
    return await _client(request).list_bindings(
        domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.put("/bindings")
async def upsert_binding(payload: MediaStudioBindingUpsertPayload, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:model_binding_manage")
    return await _client(request).upsert_binding(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
    )


@router.get("/image-generations/runs")
async def list_image_runs(request: Request, run_status: str | None = Query(default=None, alias="status"), limit: int = Query(default=50, ge=1, le=200)):
    domain_id, _, _ = await _require(request, "media_studio:image_generate")
    return await _client(request).list_image_runs(
        domain_id=domain_id, auth_context=request.state.auth_context,
        status=run_status, limit=limit,
    )


@router.post("/image-generations/runs")
async def create_image_run(payload: MediaStudioImageCreatePayload, request: Request, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
    domain_id, _, _ = await _require(request, "media_studio:image_generate")
    view = await _client(request).create_image_run(
        payload={"domain_id": domain_id, **payload.model_dump(mode="json")},
        auth_context=request.state.auth_context,
        idempotency_key=idempotency_key,
    )
    return _run_response(view)


@router.get("/image-generations/runs/{run_id}")
async def get_image_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:image_generate")
    return await _client(request).get_image_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/image-generations/runs/{run_id}/events")
async def list_image_events(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:image_generate")
    return await _client(request).list_image_events(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.delete("/image-generations/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_image_run(run_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:image_generate")
    await _client(request).delete_image_run(
        run_id=run_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/media-assets")
async def list_media_assets(request: Request, run_id: UUID | None = None, asset_status: str | None = Query(default=None, alias="status"), limit: int = Query(default=50, ge=1, le=200)):
    domain_id, _, _ = await _require(request, "media_studio:media_read")
    return await _client(request).list_media_assets(
        domain_id=domain_id, auth_context=request.state.auth_context,
        run_id=run_id, status=asset_status, limit=limit,
    )


@router.get("/media-assets/{asset_id}")
async def get_media_asset(asset_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:media_read")
    return await _client(request).get_media_asset(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/media-assets/{asset_id}/content")
async def get_media_content(asset_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:media_read")
    data, mime_type = await _client(request).get_media_content(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )
    return Response(content=data, media_type=mime_type, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@router.post("/media-assets/{asset_id}:download-url")
async def create_media_download_url(asset_id: UUID, request: Request):
    domain_id, _, _ = await _require(request, "media_studio:media_read")
    return await _client(request).create_media_download_url(
        asset_id=asset_id, domain_id=domain_id, auth_context=request.state.auth_context,
    )


@router.get("/runs")
async def list_runs(request: Request, kind: str | None = None, run_status: str | None = Query(default=None, alias="status"), limit: int = Query(default=50, ge=1, le=200)):
    domain_id, _, _ = await _require(request, "media_studio:run_read")
    return await _client(request).list_runs(
        domain_id=domain_id, auth_context=request.state.auth_context,
        kind=kind, status=run_status, limit=limit,
    )
