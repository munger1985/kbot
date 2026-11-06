"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeReferenceOccurrenceEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_REFERENCE_OCCURRENCE"
    __table_args__ = (
        Index("IX_KR_K2_6_0", "source_node_id"),
        Index("IX_KR_K2_6_1", "parse_run_id", "local_key"),
        Index("IX_KR_K2_6_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    reference_occurrence_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_KR_K2_REF_OCC_0"), nullable=False)
    source_node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_NODE.node_id", name="FK_KR_K2_REF_OCC_1"), nullable=True)
    local_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source_span_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    raw_reference: Mapped[str | None] = mapped_column(String(512), nullable=True)
    normalized_label: Mapped[str | None] = mapped_column(String(512), nullable=True)
    numbering_scope: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resolution_status: Mapped[str | None] = mapped_column(String(512), nullable=True)
    candidate_keys_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
