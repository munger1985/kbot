from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class KbEntity(BaseEntity):
    """Table for Knowledge Base configuration (Model-driven).

    This entity maps to the database table `kbot_md_kb` and stores core configuration
    for knowledge bases, including association with apps/domains, model settings,
    processing rules, security levels, and audit timestamps.
    """

    __tablename__ = "kbot_md_kb"
    
    kb_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique knowledge base identifier, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID (composes composite unique constraint with DOMAIN_ID and KB_NAME)")
    domain_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Associated business domain ID")
    kb_name: Mapped[str | None] = mapped_column(String(256), comment="Knowledge base name (unique within the same business domain)")
    kb_category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Knowledge base type enumeration")
    descs: Mapped[str | None] = mapped_column(String(512), comment="Detailed description of the knowledge base")
    txt_embed_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Text embedding model ID")
    img_embed_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Image embedding model ID")
    img2txt_model_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="Image-to-text model ID")
    llm_model_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="LLM model ID")
    kb_status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Knowledge base status enumeration")
    security_level: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="File security level enumeration")
    process_priority: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Processing priority enumeration")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")