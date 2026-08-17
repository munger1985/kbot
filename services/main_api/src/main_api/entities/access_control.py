"""平台身份、应用目录与分层授权实体。"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, ForeignKeyConstraint, Numeric, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity


class PlatformApplicationEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_APP"

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    member_assignable: Mapped[str] = mapped_column(String(1), nullable=False, default="Y")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PlatformUserEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_USER"

    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    display_name: Mapped[str | None] = mapped_column(String(256))
    account_origin: Mapped[str] = mapped_column(String(16), nullable=False, default="PLATFORM")
    owner_app_id: Mapped[str | None] = mapped_column(String(64), ForeignKey("KBOT_PLATFORM_APP.app_id"))
    is_protected: Mapped[str] = mapped_column(String(1), nullable=False, default="N")
    max_security_level: Mapped[int] = mapped_column(Numeric(3, 0), nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PlatformUserCredentialEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_USER_CREDENTIAL"

    user_id: Mapped[str] = mapped_column(String(256), ForeignKey("KBOT_PLATFORM_USER.user_id"), primary_key=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    must_change_password: Mapped[str] = mapped_column(String(1), nullable=False, default="Y")
    password_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class PermissionEntity(BaseEntity):
    __tablename__ = "KBOT_PERMISSION"
    __table_args__ = (
        UniqueConstraint("app_id", "permission_code", name="UK_PERMISSION_APP_CODE"),
    )

    permission_code: Mapped[str] = mapped_column(String(128), primary_key=True)
    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_PLATFORM_APP.app_id"), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)


class AppRoleEntity(BaseEntity):
    __tablename__ = "KBOT_APP_ROLE"

    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_PLATFORM_APP.app_id"), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    is_system: Mapped[str] = mapped_column(String(1), nullable=False, default="N")
    scope_policy: Mapped[str] = mapped_column(String(32), nullable=False, default="SELECTABLE")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)


class AppRolePermissionEntity(BaseEntity):
    __tablename__ = "KBOT_APP_ROLE_PERMISSION"
    __table_args__ = (
        ForeignKeyConstraint(["app_id", "role_code"], ["KBOT_APP_ROLE.app_id", "KBOT_APP_ROLE.role_code"], name="FK_APP_ROLE_PERMISSION_ROLE"),
        ForeignKeyConstraint(["app_id", "permission_code"], ["KBOT_PERMISSION.app_id", "KBOT_PERMISSION.permission_code"], name="FK_APP_ROLE_PERM_PERMISSION"),
    )

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    permission_code: Mapped[str] = mapped_column(String(128), primary_key=True)


class PlatformUserRoleEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_USER_ROLE"
    __table_args__ = (
        ForeignKeyConstraint(
            ["app_id", "role_code"],
            ["KBOT_APP_ROLE.app_id", "KBOT_APP_ROLE.role_code"],
            name="FK_PLATFORM_USER_ROLE_ROLE",
        ),
    )

    user_id: Mapped[str] = mapped_column(String(256), ForeignKey("KBOT_PLATFORM_USER.user_id"), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    app_id: Mapped[str] = mapped_column(String(64), nullable=False, default="platform")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AppDomainEntity(BaseEntity):
    __tablename__ = "KBOT_APP_DOMAIN"

    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_PLATFORM_APP.app_id"), primary_key=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), ForeignKey("KBOT_PLATFORM_DOMAIN.domain_id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AppMemberEntity(BaseEntity):
    __tablename__ = "KBOT_APP_MEMBER"

    app_id: Mapped[str] = mapped_column(String(64), ForeignKey("KBOT_PLATFORM_APP.app_id"), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), ForeignKey("KBOT_PLATFORM_USER.user_id"), primary_key=True)
    member_source: Mapped[str] = mapped_column(String(32), nullable=False)
    is_initial_admin: Mapped[str] = mapped_column(String(1), nullable=False, default="N")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    granted_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class AppMemberRoleEntity(BaseEntity):
    __tablename__ = "KBOT_APP_MEMBER_ROLE"
    __table_args__ = (
        ForeignKeyConstraint(["app_id", "role_code"], ["KBOT_APP_ROLE.app_id", "KBOT_APP_ROLE.role_code"], name="FK_APP_MEMBER_ROLE_ROLE"),
        ForeignKeyConstraint(["app_id", "user_id"], ["KBOT_APP_MEMBER.app_id", "KBOT_APP_MEMBER.user_id"], name="FK_APP_MEMBER_ROLE_MEMBER"),
    )

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="SELECTED_DOMAINS")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AppMemberRoleScopeEntity(BaseEntity):
    __tablename__ = "KBOT_APP_MEMBER_ROLE_SCOPE"
    __table_args__ = (
        ForeignKeyConstraint(["app_id", "user_id", "role_code"], ["KBOT_APP_MEMBER_ROLE.app_id", "KBOT_APP_MEMBER_ROLE.user_id", "KBOT_APP_MEMBER_ROLE.role_code"], name="FK_APP_MEMBER_SCOPE_BINDING"),
        ForeignKeyConstraint(["app_id", "domain_id"], ["KBOT_APP_DOMAIN.app_id", "KBOT_APP_DOMAIN.domain_id"], name="FK_APP_MEMBER_SCOPE_DOMAIN"),
    )

    app_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    role_code: Mapped[str] = mapped_column(String(64), primary_key=True)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)


__all__ = [
    "AppDomainEntity", "AppMemberEntity", "AppMemberRoleEntity",
    "AppMemberRoleScopeEntity", "AppRoleEntity", "AppRolePermissionEntity",
    "PermissionEntity", "PlatformApplicationEntity", "PlatformUserCredentialEntity",
    "PlatformUserEntity", "PlatformUserRoleEntity",
]
