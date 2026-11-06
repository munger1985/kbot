"""本 App 资料组织与显式成员授权；不复用公共 KC Collection。"""

from datetime import datetime
from uuid import UUID
from sqlalchemy import Index, Integer, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class KnowledgeLibraryEntity(BaseEntity):
    __tablename__ = "KBOT_KR_KNOWLEDGE_LIBRARY"
    __table_args__ = (Index("IX_KR_LIBRARY_DOMAIN", "domain_id", "status"),)

    library_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    grants_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    configuration_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now())
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}
