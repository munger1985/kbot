"""平台 Domain 注册实体。"""

from datetime import datetime

from sqlalchemy import DateTime, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from platform_core.persistence.orm import BaseEntity


class PlatformDomainEntity(BaseEntity):
    __tablename__ = "KBOT_PLATFORM_DOMAIN"

    domain_id: Mapped[int] = mapped_column(
        Numeric(38, 0),
        primary_key=True,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="ACTIVE",
    )
    description: Mapped[str | None] = mapped_column(String(1000))
    row_version: Mapped[int] = mapped_column(
        Numeric(19, 0),
        nullable=False,
        default=1,
    )
    created_by: Mapped[str | None] = mapped_column(String(256))
    updated_by: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


__all__ = ["PlatformDomainEntity"]
