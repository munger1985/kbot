"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeParseBatchEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_PARSE_BATCH"
    __table_args__ = (
        Index("IX_OPS_K2_3_0", "parse_run_id", "sequence"),
        Index("IX_OPS_K2_3_1", "parse_run_id", "batch_key"),
        Index("IX_OPS_K2_3_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    parse_batch_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_PARSE_BATCH_0"), nullable=False)
    batch_key: Mapped[str] = mapped_column(String(512), nullable=False)
    sequence: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    dependency_keys_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    artifact_manifest_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    event_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
