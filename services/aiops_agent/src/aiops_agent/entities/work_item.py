"""DBA 工作项、发生记录、资源关联和活动时间线映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.identity import uuid7
from platform_core.persistence.orm import (
    BaseEntity,
    OracleNativeJSON,
    UniversalTimestamp,
    UUIDv7Type,
)


class WorkItemEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORK_ITEM"

    work_item_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    item_key: Mapped[str] = mapped_column(String(32), nullable=False)
    target_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    work_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    priority: Mapped[str] = mapped_column(String(4), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    phase: Mapped[str] = mapped_column(String(24), nullable=False)
    wait_reason: Mapped[str | None] = mapped_column(String(256))
    responsibility_group_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    assignee_user_id: Mapped[str | None] = mapped_column(String(256))
    assignment_source: Mapped[str | None] = mapped_column(String(32))
    assigned_by: Mapped[str | None] = mapped_column(String(256))
    assigned_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    acknowledgement_due_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    resolution_due_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    verification_due_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    first_observed_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    last_observed_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    occurrence_count: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    reopen_count: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=0
    )
    resolution_code: Mapped[str | None] = mapped_column(String(32))
    resolution_note: Mapped[str | None] = mapped_column(Text)
    completion_note: Mapped[str | None] = mapped_column(Text)
    completed_by: Mapped[str | None] = mapped_column(String(256))
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    verification_result: Mapped[str | None] = mapped_column(String(16))
    verification_note: Mapped[str | None] = mapped_column(Text)
    verified_by: Mapped[str | None] = mapped_column(String(256))
    verified_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    routing_decision_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    resolved_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        UniversalTimestamp(timezone=True)
    )
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False
    )
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0), nullable=False, default=1
    )
    __mapper_args__ = {"version_id_col": row_version}


class ResponsibilityGroupEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESPONSIBILITY_GROUP"

    responsibility_group_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True, default=uuid7)
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    lead_user_id: Mapped[str | None] = mapped_column(String(256))
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    row_version: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False, default=1)
    __mapper_args__ = {"version_id_col": row_version}


class ResponsibilityGroupMemberEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_RESPONSIBILITY_GROUP_MEMBER"

    responsibility_group_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    member_role: Mapped[str] = mapped_column(String(16), nullable=False, default="MEMBER")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ACTIVE")
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    updated_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class WorkItemOccurrenceEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORK_ITEM_OCCURRENCE"

    occurrence_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    work_item_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    ops_run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    situation_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    finding_id: Mapped[str] = mapped_column(String(128), nullable=False)
    finding_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confirmation: Mapped[str] = mapped_column(String(16), nullable=False)
    finding_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(
        OracleNativeJSON
    )
    evidence_refs_json: Mapped[list[str] | None] = mapped_column(
        OracleNativeJSON
    )
    observed_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class WorkItemLinkEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORK_ITEM_LINK"

    work_item_link_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    work_item_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    resource_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    link_role: Mapped[str] = mapped_column(String(32), nullable=False)
    snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )


class WorkItemActivityEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_WORK_ITEM_ACTIVITY"

    activity_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), primary_key=True, default=uuid7
    )
    work_item_id: Mapped[UUID] = mapped_column(UUIDv7Type(), nullable=False)
    activity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str | None] = mapped_column(String(32))
    from_phase: Mapped[str | None] = mapped_column(String(24))
    to_phase: Mapped[str | None] = mapped_column(String(24))
    detail_json: Mapped[dict[str, Any] | None] = mapped_column(OracleNativeJSON)
    created_at: Mapped[datetime] = mapped_column(
        UniversalTimestamp(timezone=True), server_default=func.now(), nullable=False
    )
