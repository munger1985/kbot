"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeUnitSourceEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_UNIT_SOURCE"
    __table_args__ = (
        Index("IX_KR_K2_9_0", "search_unit_id", "unit_start"),
        Index("IX_KR_K2_9_1", "source_node_id"),
        Index("IX_KR_K2_9_2", "source_asset_id"),
        Index("IX_KR_K2_9_3", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    unit_source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    search_unit_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_SEARCH_UNIT.search_unit_id", name="FK_KR_K2_UNIT_SOURCE_0"), nullable=False)
    source_node_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_NODE.node_id", name="FK_KR_K2_UNIT_SOURCE_1"), nullable=True)
    source_asset_id: Mapped[UUID | None] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_VISUAL_ASSET.visual_asset_id", name="FK_KR_K2_UNIT_SOURCE_2"), nullable=True)
    unit_start: Mapped[int | None] = mapped_column(Numeric(19, 0), nullable=True)
    unit_end: Mapped[int | None] = mapped_column(Numeric(19, 0), nullable=True)
    source_start: Mapped[int | None] = mapped_column(Numeric(19, 0), nullable=True)
    source_end: Mapped[int | None] = mapped_column(Numeric(19, 0), nullable=True)
    locator_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
