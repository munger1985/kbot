"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeNodeEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_NODE"
    __table_args__ = (
        Index("IX_OPS_K2_4_0", "parse_run_id", "local_key"),
        Index("IX_OPS_K2_4_1", "parse_run_id", "reading_order"),
        Index("IX_OPS_K2_4_2", "parent_node_id"),
        Index("IX_OPS_K2_4_3", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    node_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_NODE_0"), nullable=False)
    local_key: Mapped[str] = mapped_column(String(512), nullable=False)
    kind: Mapped[str] = mapped_column(String(512), nullable=False)
    parent_node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_NODE.node_id", name="FK_OPS_K2_NODE_1"), nullable=True)
    reading_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    original_text_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    allowed_view_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    locator_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    provenance_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    metadata_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
