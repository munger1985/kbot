"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeRepresentationEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_REPRESENTATION"
    __table_args__ = (
        Index("IX_KR_K2_10_0", "space_id", "search_unit_id"),
        Index("IX_KR_K2_10_1", "search_unit_id"),
        Index("IX_KR_K2_10_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    representation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    search_unit_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_KR_KNOWLEDGE_SEARCH_UNIT.search_unit_id", name="FK_KR_K2_REPRESENTATION_0"), nullable=False)
    space_id: Mapped[str] = mapped_column(String(512), nullable=False)
    model_snapshot_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    shape_json: Mapped[dict | list] = mapped_column(OracleNativeJSON, nullable=False)
    mask_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    vector_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tensor_ref: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
