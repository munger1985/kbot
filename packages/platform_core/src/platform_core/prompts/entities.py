"""Platform Prompt Registry 的 SQLAlchemy 映射。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UUIDv7Type,
)


class PlatformPromptEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_PROMPT"

    prompt_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    prompt_key: Mapped[str] = mapped_column(
        String(256), nullable=False, unique=True
    )
    owner_service: Mapped[str] = mapped_column(String(64), nullable=False)
    purpose: Mapped[str] = mapped_column(String(1000), nullable=False)
    active_version_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    __mapper_args__ = {"version_id_col": row_version}


class PlatformPromptVersionEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_PROMPT_VERSION"

    prompt_version_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    prompt_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_variables_json: Mapped[list[str]] = mapped_column(
        OracleNativeJSON, nullable=False
    )
    output_schema_ref: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
