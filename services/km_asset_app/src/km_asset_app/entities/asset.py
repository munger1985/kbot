"""KM Asset 来源、快照、附件和持久任务实体。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UUIDv7Type


class KmSourceEntity(BaseEntity):
    __tablename__ = "KBOT_KM_SOURCE"
    __table_args__ = (
        UniqueConstraint("domain_id", "display_name", name="UK_KM_SOURCE_NAME"),
        Index("IX_KM_SOURCE_SCOPE", "domain_id", "status"),
    )

    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    metadb_endpoint: Mapped[str] = mapped_column(String(2048), nullable=False)
    metadb_credential_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    sharepoint_credential_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    sharepoint_site_path: Mapped[str] = mapped_column(String(512), nullable=False)
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    semantic_model_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    policy_binding_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    model_catalog_hash: Mapped[str | None] = mapped_column(String(64))
    model_status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT")
    poll_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class KmAssetEntity(BaseEntity):
    __tablename__ = "KBOT_KM_ASSET"
    __table_args__ = (
        UniqueConstraint("source_id", "external_asset_id", name="UK_KM_ASSET_SOURCE"),
        Index("IX_KM_ASSET_SCOPE_STATUS", "domain_id", "ingestion_status", "synced_at"),
        Index("IX_KM_ASSET_AUTHOR", "domain_id", "author_mail"),
    )

    km_asset_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    external_asset_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_revision: Mapped[str | None] = mapped_column(String(256))
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_status: Mapped[str] = mapped_column(String(16), nullable=False)
    ingestion_status: Mapped[str] = mapped_column(String(24), nullable=False)
    asset_title: Mapped[str | None] = mapped_column(String(512))
    author_mail: Mapped[str | None] = mapped_column(String(512))
    asset_product: Mapped[str | None] = mapped_column(String(512))
    asset_solution: Mapped[str | None] = mapped_column(String(1000))
    industry_id: Mapped[str | None] = mapped_column(String(512))
    content_category: Mapped[str | None] = mapped_column(String(512))
    asset_status: Mapped[str | None] = mapped_column(String(128))
    publish_date: Mapped[str | None] = mapped_column(String(128))
    last_update_time: Mapped[str | None] = mapped_column(String(128))
    raw_metadata_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    normalized_metadata_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    current_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    kc_bundle_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    kc_bundle_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    failure_stage: Mapped[str | None] = mapped_column(String(32))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class KmAssetRevisionEntity(BaseEntity):
    __tablename__ = "KBOT_KM_ASSET_REVISION"
    __table_args__ = (
        UniqueConstraint("km_asset_id", "revision_no", name="UK_KM_ASSET_REV_NO"),
        UniqueConstraint("km_asset_id", "source_revision", name="UK_KM_ASSET_REV_SOURCE"),
    )

    asset_revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    km_asset_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False)
    source_revision: Mapped[str] = mapped_column(String(256), nullable=False)
    snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    normalized_payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    kc_bundle_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class KmAttachmentEntity(BaseEntity):
    __tablename__ = "KBOT_KM_ATTACHMENT"
    __table_args__ = (UniqueConstraint("asset_revision_id", "external_document_id", name="UK_KM_ATTACHMENT_SOURCE"),)

    attachment_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    asset_revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    external_document_id: Mapped[str] = mapped_column(String(256), nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    file_name: Mapped[str | None] = mapped_column(String(512))
    mime_type: Mapped[str | None] = mapped_column(String(255))
    ordinal_no: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_size: Mapped[int | None] = mapped_column(Numeric(19, 0))
    content_sha256: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class KmJobEntity(BaseEntity):
    __tablename__ = "KBOT_KM_JOB"
    __table_args__ = (
        UniqueConstraint("domain_id", "idempotency_key", name="UK_KM_JOB_KEY"),
        Index("IX_KM_JOB_CLAIM", "status", "available_at", "priority"),
    )

    job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    source_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    km_asset_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    asset_revision_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON(), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(256))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
