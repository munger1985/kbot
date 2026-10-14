"""业务系统、资源与多参与者拓扑的 Oracle 映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


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


class ManagedResourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_MANAGED_RESOURCE"

    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    resource_code: Mapped[str | None] = mapped_column(String(64))
    resource_type_key: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    identity_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="STABLE")
    lifecycle_status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    owner_group_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    location_name: Mapped[str | None] = mapped_column(String(256))
    identity_fingerprint: Mapped[str | None] = mapped_column(String(64))
    attributes_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class DatabaseResourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_DATABASE_RESOURCE"

    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)


class ComputeResourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_COMPUTE_RESOURCE"

    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    compute_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    platform_name: Mapped[str | None] = mapped_column(String(128))
    platform_version: Mapped[str | None] = mapped_column(String(128))


class BusinessSystemResourceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM_RESOURCE"

    system_resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    architecture_layer_key: Mapped[str] = mapped_column(String(32), nullable=False)
    responsibility_group_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    display_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class ResourceRelationEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESOURCE_RELATION"

    relation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    business_system_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    relation_type_key: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(256))
    responsibility_group_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    origin: Mapped[str] = mapped_column(String(16), nullable=False, default="MANUAL")
    confirmation_status: Mapped[str] = mapped_column(String(16), nullable=False, default="CONFIRMED")
    verification_status: Mapped[str] = mapped_column(String(16), nullable=False, default="UNKNOWN")
    attributes_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    __mapper_args__ = {"version_id_col": row_version}


class RelationParticipantEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RELATION_PARTICIPANT"

    participant_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    relation_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    system_resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    participant_role: Mapped[str] = mapped_column(String(64), nullable=False)
    display_order: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=0)


class BusinessSystemDraftEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM_DRAFT"

    draft_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    business_system_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    mode: Mapped[str] = mapped_column("draft_mode", String(16), nullable=False)
    current_step: Mapped[str] = mapped_column(String(32), nullable=False, default="BUSINESS_INFO")
    revision: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    base_row_version: Mapped[int | None] = mapped_column(Numeric(19, 0))
    system_payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    expires_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class BusinessSystemDraftChangeEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_BUSINESS_SYSTEM_DRAFT_CHANGE"

    draft_change_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    draft_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    sequence_no: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    change_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)


class ResourceDiscoveryJobEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESOURCE_DISCOVERY_JOB"

    discovery_job_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    draft_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    connector_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    credential_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    request_json: Mapped[dict[str, Any]] = mapped_column(OracleNativeJSON, nullable=False, default=dict)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    error_code: Mapped[str | None] = mapped_column(String(128))
    started_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
