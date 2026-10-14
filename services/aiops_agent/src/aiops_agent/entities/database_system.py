"""数据库系统及结构化拓扑的 Oracle 映射。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class DatabaseSystemEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DATABASE_SYSTEM"

    database_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    db_type: Mapped[str] = mapped_column(String(32), nullable=False)
    version_code: Mapped[str] = mapped_column(String(64), nullable=False)
    deployment_shape: Mapped[str] = mapped_column(String(32), nullable=False)
    deployment_category: Mapped[str] = mapped_column(String(32), nullable=False)
    capabilities_json: Mapped[list[str]] = mapped_column(OracleNativeJSON, nullable=False, default=list)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    importance_level: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=3)
    responsibility_group_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class DatabaseSystemSiteEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DATABASE_SYSTEM_SITE"

    site_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    database_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    site_code: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    expected_role: Mapped[str] = mapped_column(String(16), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)


class DatabaseSystemMemberEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DATABASE_SYSTEM_MEMBER"

    member_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    database_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    site_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    member_key: Mapped[str] = mapped_column(String(64), nullable=False)
    group_key: Mapped[str | None] = mapped_column(String(64))
    member_type: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_role: Mapped[str] = mapped_column(String(32), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)


class DatabaseSystemRelationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DATABASE_SYSTEM_RELATION"

    relation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    database_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_member_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_member_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    relation_key: Mapped[str] = mapped_column(String(64), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    transport_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)
