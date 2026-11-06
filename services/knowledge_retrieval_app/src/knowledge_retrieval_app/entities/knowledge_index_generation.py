"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeIndexGenerationEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_INDEX_GENERATION"
    __table_args__ = (
        Index("IX_KR_K2_11_0", "document_version_id", "status"),
        Index("IX_KR_K2_11_1", "parse_run_id", "projection_ref"),
        Index("IX_KR_K2_11_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    index_generation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_KR_K2_INDEX_GENERATION_0"), nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_KR_K2_INDEX_GENERATION_1"), nullable=False)
    projection_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    manifest_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    channel_indexes_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    active_revision: Mapped[int | None] = mapped_column(Numeric(19, 0), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
