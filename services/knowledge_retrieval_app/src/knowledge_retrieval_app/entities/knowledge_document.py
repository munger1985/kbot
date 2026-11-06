"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeDocumentEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_DOCUMENT"
    __table_args__ = (
        Index("IX_KR_K2_0_0", "domain_id", "status"),
        Index("IX_KR_K2_0_1", "business_asset_ref"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    document_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    business_asset_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    metadata_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
