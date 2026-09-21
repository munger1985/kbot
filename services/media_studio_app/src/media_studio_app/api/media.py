"""多媒体创作工作台图片资产内部 API。"""

from uuid import UUID

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from media_studio_app.api.context import raise_application_error, require_actor
from media_studio_app.application import MediaStudioApplicationError, MediaAssetService


router = APIRouter(prefix="/internal/v1/media-studio/media-assets", tags=["MediaStudio Media"])


def _service(request: Request) -> MediaAssetService:
    return request.app.state.media_service


@router.get("")
async def list_media_assets(
    domain_id: int,
    request: Request,
    run_id: UUID | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    require_actor(request, domain_id)
    return await _service(request).list(
        domain_id=domain_id, run_id=run_id, status=status, limit=limit,
    )


@router.get("/{asset_id}")
async def get_media_asset(asset_id: UUID, domain_id: int, request: Request):
    require_actor(request, domain_id)
    try:
        return await _service(request).get(domain_id=domain_id, asset_id=asset_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)


@router.get("/{asset_id}/content")
async def get_media_content(asset_id: UUID, domain_id: int, request: Request):
    require_actor(request, domain_id)
    try:
        data, mime_type = await _service(request).content(domain_id=domain_id, asset_id=asset_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)
    return Response(
        content=data,
        media_type=mime_type,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{asset_id}:download-url")
async def create_download_url(asset_id: UUID, domain_id: int, request: Request):
    require_actor(request, domain_id)
    try:
        await _service(request).get(domain_id=domain_id, asset_id=asset_id)
    except MediaStudioApplicationError as exc:
        raise_application_error(exc)
    return _service(request).download_url(asset_id)
