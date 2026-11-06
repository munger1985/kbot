"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeRetrievalReceiptEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_RETRIEVAL_RECEIPT"
    __table_args__ = (
        Index("IX_KR_K2_14_0", "subject_ref", "request_id"),
        Index("IX_KR_K2_14_1", "scope_snapshot_id"),
        Index("IX_KR_K2_14_2", "expires_at"),
        Index("IX_KR_K2_14_3", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    retrieval_receipt_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    request_id: Mapped[str] = mapped_column(String(512), nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    scope_snapshot_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_SCOPE_SNAPSHOT.scope_snapshot_id", name="FK_KR_K2_RETRIEVAL_RECEIPT_0"), nullable=True)
    snapshot_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    query_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    plan_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    evidence_manifest_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    coverage_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    cursor_state_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
