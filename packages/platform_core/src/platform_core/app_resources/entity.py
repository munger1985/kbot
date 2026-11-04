"""案例资源初始化记录；只存稳定身份、状态和重试信息，不保存凭证。"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column
from platform_core.persistence.orm import BaseEntity, OracleNativeJSON, UUIDv7Type, UniversalTimestamp


class CaseResourceEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_CASE_RESOURCE"

    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(256), nullable=False)
    collection_id: Mapped[UUID | None] = mapped_column(UUIDv7Type())
    creation_allowed: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    configuration_status: Mapped[str] = mapped_column(String(32), nullable=False)
    missing_model_roles: Mapped[list] = mapped_column(OracleNativeJSON, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False)
    next_attempt_at: Mapped[datetime] = mapped_column(UniversalTimestamp(timezone=True), nullable=False, index=True)
    lease_token: Mapped[str | None] = mapped_column(String(36))
    lease_until: Mapped[datetime | None] = mapped_column(UniversalTimestamp(timezone=True))
