"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeRelationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_RELATION"
    __table_args__ = (
        Index("IX_OPS_K2_7_0", "source_node_id", "relation_type"),
        Index("IX_OPS_K2_7_1", "target_node_id", "relation_type"),
        Index("IX_OPS_K2_7_2", "target_asset_id"),
        Index("IX_OPS_K2_7_3", "occurrence_id"),
        Index("IX_OPS_K2_7_4", "parse_run_id"),
        Index("IX_OPS_K2_7_5", "target_document_version_id"),
        Index("IX_OPS_K2_7_6", "target_parse_run_id"),
        Index("IX_OPS_K2_7_7", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    relation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_RELATION_0"), nullable=False)
    source_node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_NODE.node_id", name="FK_OPS_K2_RELATION_1"), nullable=True)
    target_node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_NODE.node_id", name="FK_OPS_K2_RELATION_2"), nullable=True)
    target_asset_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_VISUAL_ASSET.visual_asset_id", name="FK_OPS_K2_RELATION_3"), nullable=True)
    target_document_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_DOCUMENT_VERSION.document_version_id", name="FK_OPS_K2_RELATION_4"), nullable=True)
    target_parse_run_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_RELATION_5"), nullable=True)
    relation_type: Mapped[str | None] = mapped_column(String(512), nullable=True)
    occurrence_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_REFERENCE_OCCURRENCE.reference_occurrence_id", name="FK_OPS_K2_RELATION_6"), nullable=True)
    basis_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
