"""AIOps 应用拥有的 Agent、不可变版本和授权。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Index, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class AIOpsAgentEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_AGENT"
    __table_args__ = (
        UniqueConstraint("domain_id", "display_name", name="UK_OPS_AGENT_NAME"),
        Index("IX_OPS_AGENT_SCOPE_STATUS", "domain_id", "status"),
    )

    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT")
    current_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class AIOpsAgentVersionEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_AGENT_VERSION"
    __table_args__ = (
        UniqueConstraint("agent_id", "version_no", name="UK_OPS_AGENT_VERSION"),
    )

    agent_version_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False, index=True)
    version_no: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False)
    monitor_source_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    policy_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    inspection_plan_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    models_json: Mapped[dict[str, str]] = mapped_column(OracleNativeJSON, nullable=False)
    image_capabilities_json: Mapped[dict[str, Any]] = mapped_column(
        OracleNativeJSON, nullable=False, default=dict
    )
    instruction: Mapped[str | None] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class AIOpsAgentGrantEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_AGENT_GRANT"
    __table_args__ = (
        UniqueConstraint(
            "agent_id", "subject_type", "subject_id", name="UK_OPS_AGENT_GRANT_SUBJECT"
        ),
        Index(
            "IX_OPS_AGENT_GRANT_SCOPE",
            "domain_id",
            "subject_type",
            "subject_id",
            "status",
        ),
    )

    agent_grant_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    subject_type: Mapped[str] = mapped_column(String(16), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
