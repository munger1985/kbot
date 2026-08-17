"""单 Domain 应用成员与权限实体。"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity


class PlatformUserEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_USER"

    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(256))
    max_security_level: Mapped[int] = mapped_column(
        Numeric(3, 0), nullable=False, default=1
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PlatformUserCredentialEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_USER_CREDENTIAL"

    user_id: Mapped[str] = mapped_column(
        String(256),
        ForeignKey("KBOT_PLATFORM_USER.user_id"),
        primary_key=True,
    )
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    must_change_password: Mapped[str] = mapped_column(
        String(1), nullable=False, default="Y"
    )
    password_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PermissionEntity(BaseEntity):
    __tablename__ = "KBOT_PERMISSION"

    permission_code: Mapped[str] = mapped_column(String(128), primary_key=True)
    app_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)


class AppRoleEntity(BaseEntity):
    __tablename__ = "KBOT_APP_ROLE"

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")


class AppRolePermissionEntity(BaseEntity):
    __tablename__ = "KBOT_APP_ROLE_PERMISSION"
    __table_args__ = (
        ForeignKeyConstraint(
            ["app_id", "role_code"],
            ["KBOT_APP_ROLE.app_id", "KBOT_APP_ROLE.role_code"],
            name="FK_APP_ROLE_PERMISSION_ROLE",
        ),
    )

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    permission_code: Mapped[str] = mapped_column(
        String(128), primary_key=True
    )


class AppMemberRoleEntity(BaseEntity):
    __tablename__ = "KBOT_APP_MEMBER_ROLE"
    __table_args__ = (
        ForeignKeyConstraint(
            ["app_id", "role_code"],
            ["KBOT_APP_ROLE.app_id", "KBOT_APP_ROLE.role_code"],
            name="FK_APP_MEMBER_ROLE_ROLE",
        ),
    )

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = [
    "AppMemberRoleEntity",
    "AppRoleEntity",
    "AppRolePermissionEntity",
    "PermissionEntity",
    "PlatformUserCredentialEntity",
    "PlatformUserEntity",
]
