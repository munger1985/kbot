"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeParseRunEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_PARSE_RUN"
    __table_args__ = (
        Index("IX_OPS_K2_2_0", "document_version_id", "status"),
        Index("IX_OPS_K2_2_1", "parse_run_id", "fence"),
        Index("IX_OPS_K2_2_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    document_version_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_OPS_K2_PARSE_RUN_0"), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_revision: Mapped[str] = mapped_column(String(512), nullable=False)
    model_snapshot_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    attempt: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    fence: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    quality_report_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    profile_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    checkpoint_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
