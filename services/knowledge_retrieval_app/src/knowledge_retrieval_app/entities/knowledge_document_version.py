"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeDocumentVersionEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_DOCUMENT_VERSION"
    __table_args__ = (
        Index("IX_KR_K2_1_0", "document_id", "business_version_ref"),
        Index("IX_KR_K2_1_1", "domain_id", "status"),
        Index("IX_KR_K2_1_2", "published_parse_run_id"),
        Index("IX_KR_K2_1_3", "active_generation_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    document_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_DOCUMENT.document_id", name="FK_KR_K2_DOCUMENT_VERSION_0"), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(512), nullable=False)
    byte_count: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    source_object_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    business_version_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    published_parse_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_KR_K2_DOCUMENT_VERSION_1", use_alter=True), nullable=True)
    active_generation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_INDEX_GENERATION.index_generation_id", name="FK_KR_K2_DOCUMENT_VERSION_2", use_alter=True), nullable=True)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    source_descriptor_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    allowed_view_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    allowed_view_manifest_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
