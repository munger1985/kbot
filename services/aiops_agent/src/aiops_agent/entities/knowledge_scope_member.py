"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeScopeMemberEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_SCOPE_MEMBER"
    __table_args__ = (
        Index("IX_OPS_K2_17_0", "scope_snapshot_id", "document_version_id"),
        Index("IX_OPS_K2_17_1", "document_version_id"),
        Index("IX_OPS_K2_17_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    scope_member_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    scope_snapshot_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_SCOPE_SNAPSHOT.scope_snapshot_id", name="FK_OPS_K2_SCOPE_MEMBER_0"), nullable=True)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_OPS_K2_SCOPE_MEMBER_1"), nullable=False)
    allowed_view_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
