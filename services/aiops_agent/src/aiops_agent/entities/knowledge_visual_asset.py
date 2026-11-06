"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeVisualAssetEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_VISUAL_ASSET"
    __table_args__ = (
        Index("IX_OPS_K2_5_0", "parse_run_id", "local_key"),
        Index("IX_OPS_K2_5_1", "node_id"),
        Index("IX_OPS_K2_5_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    visual_asset_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    parse_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_PARSE_RUN.parse_run_id", name="FK_OPS_K2_VISUAL_ASSET_0"), nullable=False)
    node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_KNOWLEDGE_NODE.node_id", name="FK_OPS_K2_VISUAL_ASSET_1"), nullable=True)
    local_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(512), nullable=False)
    byte_count: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    dimensions_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    crop_transform_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    locator_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    provenance_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    metadata_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
