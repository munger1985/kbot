"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeSearchUnitEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_SEARCH_UNIT"
    __table_args__ = (
        Index("IX_KR_K2_8_0", "projection_ref", "unit_key"),
        Index("IX_KR_K2_8_1", "parse_run_id"),
        Index("IX_KR_K2_8_2", "source_node_id"),
        Index("IX_KR_K2_8_3", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    search_unit_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_KR_K2_SEARCH_UNIT_0"), nullable=False)
    projection_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    unit_key: Mapped[str] = mapped_column(String(512), nullable=False)
    source_node_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_NODE.node_id", name="FK_KR_K2_SEARCH_UNIT_1"), nullable=False)
    kind: Mapped[str] = mapped_column(String(512), nullable=False)
    allowed_text_ref: Mapped[str] = mapped_column(Text, nullable=False)
    context_ref: Mapped[str] = mapped_column(Text, nullable=False)
    allowed_view_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    allowed_view_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    profile_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
