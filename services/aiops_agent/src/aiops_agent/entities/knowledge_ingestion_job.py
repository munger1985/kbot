"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeIngestionJobEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_INGESTION_JOB"
    __table_args__ = (
        Index("IX_OPS_K2_12_0", "status", "retry_at", "lease_until"),
        Index("IX_OPS_K2_12_1", "document_version_id"),
        Index("IX_OPS_K2_12_2", "parse_run_id"),
        Index("IX_OPS_K2_12_3", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    ingestion_job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_OPS_K2_INGESTION_JOB_0"), nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_INGESTION_JOB_1"), nullable=False)
    attempt: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    fence: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(512), nullable=True)
    lease_until: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    retry_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    recovery_sequence: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(512), nullable=True)
    context_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    compute_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_event_digest: Mapped[str | None] = mapped_column(String(512), nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
