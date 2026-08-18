"""KM Asset App 内部管理 API。"""

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from km_asset_app.application import CreateSourceCommand, KmAssetApplicationError, KmAssetService
from platform_core.contracts import AuthContext, ServiceIdentity


router = APIRouter(prefix="/internal/v1/km-asset", tags=["KM Asset App"])


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceCreateRequest(_Payload):
    domain_id: int = Field(ge=1)
    display_name: str = Field(min_length=1, max_length=256)
    metadb_endpoint: AnyHttpUrl
    metadb_credentials: dict[str, str] = Field(min_length=2, max_length=2)
    sharepoint_credentials: dict[str, str] = Field(min_length=3, max_length=3)
    sharepoint_site_path: str = Field(min_length=1, max_length=512)
    collection_id: UUID
    poll_interval_seconds: int = Field(default=60, ge=10, le=86400)
    batch_size: int = Field(default=100, ge=1, le=1000)

    @field_validator("metadb_credentials")
    @classmethod
    def validate_metadb_credentials(cls, value):
        if set(value) != {"username", "password"} or any(
            not str(item).strip() for item in value.values()
        ):
            raise ValueError("MetaDB 凭据必须包含 username 和 password")
        return value

    @field_validator("sharepoint_credentials")
    @classmethod
    def validate_sharepoint_credentials(cls, value):
        if set(value) != {"tenant_id", "client_id", "client_secret"} or any(
            not str(item).strip() for item in value.values()
        ):
            raise ValueError("SharePoint 凭据字段不完整")
        return value


class SourceUpdateRequest(_Payload):
    domain_id: int = Field(ge=1)
    expected_row_version: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    metadb_endpoint: AnyHttpUrl | None = None
    metadb_credentials: dict[str, str] | None = Field(default=None, min_length=2, max_length=2)
    sharepoint_credentials: dict[str, str] | None = Field(default=None, min_length=3, max_length=3)
    sharepoint_site_path: str | None = Field(default=None, min_length=1, max_length=512)
    auto_sync_enabled: bool | None = None
    poll_interval_seconds: int | None = Field(default=None, ge=10, le=86400)
    batch_size: int | None = Field(default=None, ge=1, le=1000)

    @field_validator("metadb_credentials")
    @classmethod
    def validate_metadb_credentials(cls, value):
        if value is not None and (
            set(value) != {"username", "password"}
            or any(not str(item).strip() for item in value.values())
        ):
            raise ValueError("MetaDB 凭据必须包含 username 和 password")
        return value

    @field_validator("sharepoint_credentials")
    @classmethod
    def validate_sharepoint_credentials(cls, value):
        if value is not None and (
            set(value) != {"tenant_id", "client_id", "client_secret"}
            or any(not str(item).strip() for item in value.values())
        ):
            raise ValueError("SharePoint 凭据字段不完整")
        return value

    @model_validator(mode="after")
    def require_change(self):
        values = self.model_dump(exclude={"domain_id", "expected_row_version"}, exclude_none=True)
        if not values:
            raise ValueError("至少提供一个需要修改的来源字段")
        return self


class VersionRequest(_Payload):
    domain_id: int = Field(ge=1)
    expected_row_version: int = Field(ge=1)


class RawAssetRequest(_Payload):
    domain_id: int = Field(ge=1)
    source_id: UUID
    payload: dict


def _service(request: Request) -> KmAssetService:
    return request.app.state.km_asset_service


def _context(request: Request, domain_id: int, scope: str = "km_asset.manage") -> str:
    identity = getattr(request.state, "service_identity", None)
    if not isinstance(identity, ServiceIdentity) or scope not in identity.scopes:
        raise HTTPException(403, {"code": "SERVICE_SCOPE_DENIED"})
    context = getattr(request.state, "auth_context", None)
    if not isinstance(context, AuthContext) or context.domain_id is None:
        raise HTTPException(403, {"code": "DOMAIN_CONTEXT_REQUIRED"})
    if int(context.domain_id) != domain_id:
        raise HTTPException(403, {"code": "DOMAIN_SCOPE_MISMATCH"})
    return context.asserted_user_id or context.client_id


def _raise(exc: KmAssetApplicationError) -> None:
    raise HTTPException(exc.status_code, {"code": exc.code, "message": exc.message}) from exc


@router.get("/sources")
async def list_sources(domain_id: int, request: Request):
    _context(request, domain_id)
    return await _service(request).list_sources(domain_id=domain_id)


@router.post("/sources", status_code=201)
async def create_source(payload: SourceCreateRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await _service(request).create_source(CreateSourceCommand(actor_id=actor_id, **payload.model_dump(mode="python")))
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.patch("/sources/{source_id}")
async def update_source(source_id: UUID, payload: SourceUpdateRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        changes = payload.model_dump(
            mode="python",
            exclude={"domain_id", "expected_row_version"},
            exclude_none=True,
        )
        return await _service(request).update_source(
            domain_id=payload.domain_id,
            source_id=source_id,
            expected_row_version=payload.expected_row_version,
            changes=changes,
            actor_id=actor_id,
        )
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/sources/{source_id}/activate")
async def activate_source(source_id: UUID, payload: VersionRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await _service(request).activate_source(domain_id=payload.domain_id, source_id=source_id, expected_row_version=payload.expected_row_version, actor_id=actor_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/sources/{source_id}/sync", status_code=202)
async def sync_source(source_id: UUID, domain_id: int, request: Request):
    actor_id = _context(request, domain_id)
    try:
        return await _service(request).submit_sync(domain_id=domain_id, source_id=source_id, actor_id=actor_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.get("/sources/{source_id}/metadb/assets")
async def list_metadb_assets(source_id: UUID, domain_id: int, request: Request, processed: Literal["N", "Y", "F"] = "N", offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
    _context(request, domain_id)
    try:
        return await _service(request).list_metadb_assets(domain_id=domain_id, source_id=source_id, processed=processed, offset=offset, limit=limit)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/sources/{source_id}/metadb/assets/{external_asset_id}/retry", status_code=202)
async def retry_metadb_asset(source_id: UUID, external_asset_id: str, domain_id: int, request: Request):
    actor_id = _context(request, domain_id)
    try:
        return await _service(request).retry_metadb_asset(
            domain_id=domain_id,
            source_id=source_id,
            external_asset_id=external_asset_id,
            actor_id=actor_id,
        )
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.get("/sources/{source_id}/data-model")
async def get_data_model(source_id: UUID, domain_id: int, request: Request):
    _context(request, domain_id)
    try:
        return await _service(request).managed_model(domain_id=domain_id, source_id=source_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/sources/{source_id}/data-model/reconcile")
async def reconcile_data_model(source_id: UUID, domain_id: int, request: Request):
    actor_id = _context(request, domain_id)
    try:
        return await _service(request).reconcile_model(domain_id=domain_id, source_id=source_id, actor_id=actor_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/assets/snapshots", status_code=201)
async def persist_raw_asset(payload: RawAssetRequest, request: Request):
    actor_id = _context(request, payload.domain_id, "km_asset.worker")
    try:
        return await _service(request).ingest_raw_asset(domain_id=payload.domain_id, source_id=payload.source_id, payload=payload.payload, actor_id=actor_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.get("/assets")
async def list_assets(domain_id: int, request: Request, source_id: UUID | None = None, ingestion_status: Literal["DISCOVERED", "SYNC_PENDING", "DOWNLOADING", "DOWNLOAD_FAILED", "READY_FOR_INGESTION", "INGESTING", "KC_ACCEPTED", "PARSING", "READY", "RETRY_PENDING", "FAILED"] | None = None, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)):
    _context(request, domain_id)
    return await _service(request).list_assets(domain_id=domain_id, source_id=source_id, ingestion_status=ingestion_status, offset=offset, limit=limit)


@router.get("/assets/{km_asset_id}")
async def get_asset(km_asset_id: UUID, domain_id: int, request: Request):
    _context(request, domain_id)
    try:
        return await _service(request).get_asset(domain_id=domain_id, km_asset_id=km_asset_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/assets/{km_asset_id}/retry", status_code=202)
async def retry_asset(km_asset_id: UUID, payload: VersionRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await _service(request).retry_asset(domain_id=payload.domain_id, km_asset_id=km_asset_id, expected_row_version=payload.expected_row_version, actor_id=actor_id)
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.post("/assets/{km_asset_id}/reindex", status_code=202)
async def reindex_asset(km_asset_id: UUID, payload: VersionRequest, request: Request):
    actor_id = _context(request, payload.domain_id)
    try:
        return await _service(request).reindex_asset(
            domain_id=payload.domain_id,
            km_asset_id=km_asset_id,
            expected_row_version=payload.expected_row_version,
            actor_id=actor_id,
        )
    except KmAssetApplicationError as exc:
        _raise(exc)


@router.get("/jobs")
async def list_jobs(domain_id: int, request: Request, source_id: UUID | None = None, limit: int = Query(default=100, ge=1, le=500)):
    _context(request, domain_id)
    return await _service(request).list_jobs(domain_id=domain_id, source_id=source_id, limit=limit)


@router.get("/jobs/processing")
async def list_processing_jobs(domain_id: int, request: Request, source_id: UUID | None = None, limit: int = Query(default=500, ge=1, le=2000)):
    _context(request, domain_id)
    try:
        return await _service(request).list_processing_jobs(
            domain_id=domain_id,
            source_id=source_id,
            limit=limit,
        )
    except KmAssetApplicationError as exc:
        _raise(exc)
