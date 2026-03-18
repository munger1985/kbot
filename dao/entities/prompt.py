from sqlalchemy import String, Date, CLOB, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class PromptEntity(BaseEntity):
    """Table for Prompt configuration (Model-driven).

    This entity maps to the database table `kbot_md_prompt` and stores core configuration
    for AI prompts, including identification, categorization, templates, status control,
    and audit timestamps for model-driven AI applications.
    """

    __tablename__ = "kbot_md_prompt"

    prompt_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique prompt identifier, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    domain_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated business domain ID (optional)")
    name: Mapped[str | None] = mapped_column(String(256), comment="Prompt name")
    prompt_unique_name: Mapped[str] = mapped_column(String(256), comment="Unique prompt name in database (used for model calls in programs)")
    prompt_category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Prompt category enumeration")
    template: Mapped[str | None] = mapped_column(CLOB, comment="Prompt template content (CLOB large text)")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Prompt status: 1 - Enabled, 0 - Disabled")
    descs: Mapped[str | None] = mapped_column(String(512), comment="Detailed prompt description")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")