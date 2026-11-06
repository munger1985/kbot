"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeEvidencePinEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_EVIDENCE_PIN"
    __table_args__ = (
        Index("IX_OPS_K2_15_0", "owner_run_ref"),
        Index("IX_OPS_K2_15_1", "parse_run_id"),
        Index("IX_OPS_K2_15_2", "expires_at", "released_at"),
        Index("IX_OPS_K2_15_3", "index_generation_id"),
        Index("IX_OPS_K2_15_4", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    evidence_pin_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    owner_run_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    evidence_key_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_EVIDENCE_PIN_0"), nullable=False)
    projection_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    index_generation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_INDEX_GENERATION.index_generation_id", name="FK_OPS_K2_EVIDENCE_PIN_1"), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    released_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
