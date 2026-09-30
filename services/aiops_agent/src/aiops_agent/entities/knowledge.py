"""AIOps 运维知识资产 Registry 的 Oracle 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Index, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class OperationsKnowledgeAssetEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_ASSET"
    __table_args__ = (
        Index(
            "IX_OPS_KNOW_ASSET_SCOPE",
            "domain_id", "asset_kind", "status", "updated_at",
        ),
    )

    asset_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    asset_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    current_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    security_level: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}


class OperationsKnowledgeVersionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_VERSION"
    __table_args__ = (
        Index("IX_OPS_KNOW_VER_ASSET", "asset_id", "version_no"),
        Index("IX_OPS_KNOW_VER_STATUS", "status", "created_at"),
    )

    asset_version_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    asset_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    version_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_schema_version: Mapped[str | None] = mapped_column(String(64))
    profile_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    extraction_warnings_json: Mapped[list[Any]] = mapped_column(
        OracleNativeJSON, nullable=False, default=list
    )
    published_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    retired_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}


class OperationsKnowledgeScopeEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_SCOPE"
    __table_args__ = (
        Index(
            "IX_OPS_KNOW_SCOPE_LOOKUP",
            "scope_kind", "normalized_value", "asset_version_id",
        ),
        Index("IX_OPS_KNOW_SCOPE_VER", "asset_version_id"),
    )

    scope_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    asset_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    scope_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_value: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(512), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_locator_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OperationsKnowledgeSourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_SOURCE"
    __table_args__ = (
        Index("IX_OPS_KNOW_SOURCE_VER", "asset_version_id"),
        Index("IX_OPS_KNOW_SOURCE_REPORT", "source_report_id"),
        Index("IX_OPS_KNOW_SOURCE_RUN", "source_run_id"),
        Index("IX_OPS_KNOW_SOURCE_ART", "source_artifact_id"),
    )

    source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    asset_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_report_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_artifact_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    source_external_id: Mapped[str | None] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_locator_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OperationsKnowledgeIndexEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_INDEX"
    __table_args__ = (
        Index("IX_OPS_KNOW_INDEX_VER", "asset_version_id"),
        Index("IX_OPS_KNOW_INDEX_STATUS", "index_status", "updated_at"),
    )

    index_ref_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    asset_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    bundle_revision_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    index_status: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_row_version: Mapped[int | None] = mapped_column(Numeric(19, 0))
    last_checked_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_summary: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )


class OperationsKnowledgeReviewEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_REVIEW"
    __table_args__ = (
        Index("IX_OPS_KNOW_REVIEW_QUEUE", "after_status", "created_at"),
        Index("IX_OPS_KNOW_REVIEW_VER", "asset_version_id"),
    )

    review_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    asset_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(256), nullable=False)
    comment_text: Mapped[str | None] = mapped_column(String(2000))
    before_status: Mapped[str] = mapped_column(String(32), nullable=False)
    after_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
