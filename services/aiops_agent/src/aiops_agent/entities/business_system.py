"""业务系统、可管理组件与资源拓扑的 Oracle 映射。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, UniversalTimestamp, UUIDv7Type


class BusinessSystemEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM"

    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    system_code: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    environment: Mapped[str] = mapped_column(String(16), nullable=False)
    importance_level: Mapped[int] = mapped_column(Numeric(1, 0), nullable=False, default=3)
    responsibility_group_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="DRAFT")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class ManagedComponentEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_MANAGED_COMPONENT"

    component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    component_key: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    component_type: Mapped[str] = mapped_column(String(32), nullable=False)
    responsibility_group_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class BusinessSystemComponentEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM_COMPONENT"

    system_component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)


class BusinessSystemSiteEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM_SITE"

    site_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    site_code: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    site_role: Mapped[str] = mapped_column(String(16), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(10, 0), nullable=False, default=0)


class ComponentMembershipEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_COMPONENT_MEMBERSHIP"

    membership_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    site_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    member_key: Mapped[str] = mapped_column(String(64), nullable=False)
    cluster_group_key: Mapped[str | None] = mapped_column(String(64))
    expected_role: Mapped[str] = mapped_column(String(32), nullable=False)
    sort_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)


class ResourceRelationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESOURCE_RELATION"

    relation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    target_component_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    source_membership_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    target_membership_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    relation_key: Mapped[str] = mapped_column(String(64), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    transport_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    responsibility_group_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    sort_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)
