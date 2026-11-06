"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeOutboxEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_OUTBOX"
    __table_args__ = (
        Index("IX_KR_K2_13_0", "status", "retry_at"),
        Index("IX_KR_K2_13_1", "document_version_id", "idempotency_key"),
        Index("IX_KR_K2_13_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    outbox_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_KR_K2_OUTBOX_0"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(512), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False)
    payload_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    retry_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    delivered_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(512), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    attempt: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
