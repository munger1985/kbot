"""每个数据库资源至多一份独立数据保护配置。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Numeric, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type


class ProtectionPlanEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_PROTECTION_PLAN"

    resource_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), ForeignKey("KBOT_OPS_MANAGED_RESOURCE.resource_id",
                                 name="FK_OPS_PROT_PLAN_RES"), primary_key=True,
    )
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False)
    revision: Mapped[int] = mapped_column(Numeric(19, 0), nullable=False)
    settings_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    approval_json: Mapped[dict | None] = mapped_column(OracleNativeJSON)
    next_backup_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    next_drill_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))


class ProtectionRunEntity(BaseEntity):
    """每次触发及其证据；active_key 在进程状态不明时继续占用。"""

    __tablename__ = "KBOT_OPS_PROTECTION_RUN"
    __table_args__ = (
        Index("IX_OPS_PROT_RUN_PENDING", "status", "deadline_at"),
    )

    run_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    resource_id: Mapped[UUID] = mapped_column(
        UUIDv7Type(), ForeignKey("KBOT_OPS_PROTECTION_PLAN.resource_id", name="FK_OPS_PROT_RUN_PLAN"), nullable=False,
    )
    operation: Mapped[str] = mapped_column(String(8), nullable=False)
    scheduled_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    deadline_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
    active_key: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    snapshot_json: Mapped[dict] = mapped_column(OracleNativeJSON, nullable=False)
    result_json: Mapped[dict | None] = mapped_column(OracleNativeJSON)
    report_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
