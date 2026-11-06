"""本 App 独占知识数据映射；不提供跨 App 共享表。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeScopeSnapshotEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_KNOWLEDGE_SCOPE_SNAPSHOT"
    __table_args__ = (
        Index("IX_OPS_K2_16_0", "subject_ref", "scope_hash"),
        Index("IX_OPS_K2_16_1", "expires_at", "status"),
        Index("IX_OPS_K2_16_2", "domain_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    scope_snapshot_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7, nullable=False)
    subject_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    scope_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    authorization_revision: Mapped[str] = mapped_column(String(512), nullable=False)
    policy_json: Mapped[dict | list | None] = mapped_column(OracleNativeJSON, nullable=True)
    allowed_versions_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(512), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), default=1, nullable=False)
    __mapper_args__ = {"version_id_col": row_version}
