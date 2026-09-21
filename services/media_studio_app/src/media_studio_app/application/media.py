"""文生图媒体资产查询与字节读取。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from media_studio_app.application.errors import MediaStudioApplicationError
from media_studio_app.application.runtime import media_view, not_found, run_view
from platform_core.contracts import PUBLIC_API_V1


class MediaAssetService:
    def __init__(self, *, uow_factory, object_store):
        self._uow_factory = uow_factory
        self._store = object_store

    async def list(
        self,
        *,
        domain_id: int,
        run_id: UUID | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.media_assets is not None
            rows = await uow.media_assets.list(
                domain_id=domain_id, run_id=run_id, status=status, limit=limit,
            )
            return [self._view(row) for row in rows]

    async def get(self, *, domain_id: int, asset_id: UUID) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            assert uow.media_assets is not None
            row = await uow.media_assets.get(domain_id=domain_id, asset_id=asset_id)
            if row is None:
                not_found("ASSET")
            return self._view(row)

    async def content(self, *, domain_id: int, asset_id: UUID) -> tuple[bytes, str]:
        asset = await self.get(domain_id=domain_id, asset_id=asset_id)
        try:
            data = await self._store.get(asset["object_key"])
        except FileNotFoundError as exc:
            raise MediaStudioApplicationError(
                "ASSET_OBJECT_MISSING", "图片对象不存在于本地存储", status_code=404,
            ) from exc
        return data, str(asset["mime_type"])

    @staticmethod
    def download_url(asset_id: UUID) -> dict[str, str]:
        return {
            "url": f"{PUBLIC_API_V1}/apps/media-studio/media-assets/{asset_id}/content",
        }

    @staticmethod
    def _view(row) -> dict[str, Any]:
        return media_view(
            row,
            content_path=f"{PUBLIC_API_V1}/apps/media-studio/media-assets/{row.asset_id}/content",
        )


class MediaStudioRunQueryService:
    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def list(
        self,
        *,
        domain_id: int,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            rows = await uow.runs.list(
                domain_id=domain_id, kind=kind, status=status, limit=limit,
            )
            return [run_view(row) for row in rows]
