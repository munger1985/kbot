"""App 独立 API Client、Credential 与授权范围实体。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, UUIDv7Type


class AppApiClientEntity(BaseEntity):
    __tablename__ = "KBOT_APP_API_CLIENT"
    __table_args__ = (
        ForeignKeyConstraint(
            ["app_id", "domain_id"],
            ["KBOT_APP_DOMAIN.app_id", "KBOT_APP_DOMAIN.domain_id"],
            name="FK_APP_API_CLIENT_DOMAIN",
        ),
        ForeignKeyConstraint(
            ["app_id", "subject_user_id"],
            ["KBOT_APP_MEMBER.app_id", "KBOT_APP_MEMBER.user_id"],
            name="FK_APP_API_CLIENT_MEMBER",
        ),
    )

    client_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    app_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    subject_user_id: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    rate_limit_per_minute: Mapped[int] = mapped_column(
        Numeric(10, 0), nullable=False, default=60
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )


class AppApiCredentialEntity(BaseEntity):
    __tablename__ = "KBOT_APP_API_CREDENTIAL"
    __table_args__ = (
        UniqueConstraint("public_key_id", name="UK_APP_API_CRED_PUBLIC"),
    )

    credential_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    client_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), ForeignKey("KBOT_APP_API_CLIENT.client_id"),
        nullable=False, index=True,
    )
    public_key_id: Mapped[str] = mapped_column(String(64), nullable=False)
    key_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AppApiClientScopeEntity(BaseEntity):
    __tablename__ = "KBOT_APP_API_CLIENT_SCOPE"

    client_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), ForeignKey("KBOT_APP_API_CLIENT.client_id"),
        primary_key=True,
    )
    scope_code: Mapped[str] = mapped_column(String(128), primary_key=True)


class AppApiClientAgentEntity(BaseEntity):
    __tablename__ = "KBOT_APP_API_CLIENT_AGENT"

    client_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), ForeignKey("KBOT_APP_API_CLIENT.client_id"),
        primary_key=True,
    )
    agent_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)


__all__ = [
    "AppApiClientAgentEntity",
    "AppApiClientEntity",
    "AppApiClientScopeEntity",
    "AppApiCredentialEntity",
]
