from sqlalchemy import String, Date, Numeric, func, JSON
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class KBEntity(BaseEntity):
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
    engine: Mapped[str] = mapped_column(String(100), comment="知识库解析引擎类型")
    models: Mapped[dict | None] = mapped_column(JSON, comment="知识库关联的模型配置参数")
    dbconf: Mapped[dict | None] = mapped_column(JSON, comment="知识库关联的数据库配置参数")
    kb_status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Knowledge base status enumeration")
    security_level: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="File security level enumeration")
    process_priority: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Processing priority enumeration")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")