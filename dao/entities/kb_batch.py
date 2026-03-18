from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class BatchEntity(BaseEntity):
    """Table for Knowledge Base Batch management (Model-driven).

    This entity maps to the database table `kbot_md_batch` and stores batch operation
    information for knowledge bases, including association with apps/KBs, unique constraints,
    and audit timestamps for model-driven AI applications.
    """

    __tablename__ = "kbot_md_kb_batch"
    
    batch_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique batch identifier, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    batch_name: Mapped[str] = mapped_column(String(256), nullable=False, comment="Batch name (composes composite unique constraint with KB_ID)")
    kb_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated knowledge base ID")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")