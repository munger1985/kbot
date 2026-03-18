from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class DomainEntity(BaseEntity):
    """Table for Business Domain configuration (Model-driven).

    This entity maps to the database table `kbot_md_domain` and stores configuration
    information for business domains, including association with apps, status control,
    and audit timestamps for model-driven AI applications.
    """

    __tablename__ = "kbot_md_domain"
    
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Business domain ID, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated APP ID (composes unique index with name)")
    name: Mapped[str | None] = mapped_column(String(256), comment="Business domain name")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Status: 1 - Enabled, 0 - Disabled (originally mislabeled as prompt status)")
    descs: Mapped[str | None] = mapped_column(String(512), comment="Business domain description")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")