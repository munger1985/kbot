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


class OperationsKnowledgeBaseEntity(BaseEntity):
    """Domain 内由用户管理的运维手册知识库。"""

    __tablename__ = "KBOT_OPS_KNOWLEDGE_BASE"
    __table_args__ = (
        Index("IX_OPS_KNOW_BASE_SCOPE", "domain_id", "status", "updated_at"),
    )

    knowledge_base_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    collection_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
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


class OperationsKnowledgeAssistantBindingEntity(BaseEntity):
    """Domain 内固定知识问答 Agent 的可检索知识库绑定。"""

    __tablename__ = "KBOT_OPS_KNOW_ASSISTANT_BIND"

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    assistant_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    assistant_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    composer_model_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
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


class OperationsKnowledgeAssistantBaseEntity(BaseEntity):
    """固定知识问答 Agent 与运维知识库的多对多关系。"""

    __tablename__ = "KBOT_OPS_KNOW_ASSISTANT_BASE"

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    knowledge_base_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
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
    knowledge_base_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
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
    tag_analysis_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
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
    tag_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    scope_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_value: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(512), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    source_locator_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class OperationsKnowledgeTagEntity(BaseEntity):
    """由 AIOps 控制台维护、供模型有限选择的知识标签。"""

    __tablename__ = "KBOT_OPS_KNOWLEDGE_TAG"
    __table_args__ = (
        Index("IX_OPS_KNOW_TAG_CATALOG", "domain_id", "category", "status", "sort_order"),
    )

    tag_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}


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
