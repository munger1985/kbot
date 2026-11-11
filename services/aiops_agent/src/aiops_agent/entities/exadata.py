"""Exadata 确认成员快照和只读资源诊断报告；不进入数据库任务队列。"""

from datetime import datetime
from uuid import UUID
from sqlalchemy import Numeric, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UniversalTimestamp, UUIDv7Type

JSON_TYPE = OracleNativeJSON


class ExadataInventoryEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EXADATA_INVENTORY"
    binding_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_RESOURCE_SOURCE_BINDING.resource_source_binding_id"), primary_key=True)
    inventory_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    members_json: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)
    confirmed_by: Mapped[str] = mapped_column(String(256), nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)


class ExadataReportEntity(BaseEntity):
    __tablename__ = "KBOT_OPS_EXADATA_REPORT"
    report_id: Mapped[UUID] = mapped_column(UUIDv7Type(), primary_key=True)
    resource_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_MANAGED_RESOURCE.resource_id"), nullable=False)
    binding_id: Mapped[UUID] = mapped_column(UUIDv7Type(), ForeignKey("KBOT_OPS_RESOURCE_SOURCE_BINDING.resource_source_binding_id"), nullable=False)
    report_json: Mapped[dict] = mapped_column(JSON_TYPE, nullable=False)
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False)
