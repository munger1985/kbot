"""直接检索的私有历史；只保存条件和不可变证据清单的引用。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Index, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeSearchHistoryEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_SEARCH_HISTORY"
    __table_args__ = (
        Index("IX_KR_SEARCH_HISTORY_OWNER", "domain_id", "subject_ref", "created_at", "history_id"),
        Index("IX_KR_SEARCH_HISTORY_EXPIRY", "expires_at", "history_id"),
    )

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    history_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    subject_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    library_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    conditions_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    manifest_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
