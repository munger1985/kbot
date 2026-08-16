"""知识检索应用拥有的 Agent 聚合。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, Index, Integer, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UUIDv7Type,
)


class KnowledgeRetrievalAgentEntity(BaseEntity):
    __tablename__ = "KBOT_KR_AGENT"
    __table_args__ = (
        UniqueConstraint("domain_id", "display_name", name="UK_KR_AGENT_NAME"),
        Index("IX_KR_AGENT_SCOPE_STATUS", "domain_id", "status"),
    )

    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT")
    current_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class KnowledgeRetrievalAgentVersionEntity(BaseEntity):
    __tablename__ = "KBOT_KR_AGENT_VERSION"
    __table_args__ = (
        UniqueConstraint("agent_id", "version_no", name="UK_KR_AGENT_VERSION"),
    )

    agent_version_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    enabled_capabilities_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON(), nullable=False
    )
    models_json: Mapped[dict[str, str]] = mapped_column(
        OracleNativeJSON(), nullable=False
    )
    do_rerank: Mapped[bool] = mapped_column(nullable=False, default=False)
    instruction: Mapped[str | None] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
