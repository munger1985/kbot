from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON

class FileEntity(BaseEntity):
    """Table for Knowledge Base File management (Model-driven).

    This entity maps to the database table `kbot_md_kb_files` and stores comprehensive
    metadata for files in knowledge bases, including storage info, processing status,
    security settings, parsing configurations, and audit timestamps.
    """

    __tablename__ = "kbot_md_kb_files"
    
    file_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique file identifier (UUID), primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    kb_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated knowledge base ID")
    batch: Mapped[str] = mapped_column(String(100), nullable=False, comment="Upload batch")
    file_path: Mapped[str] = mapped_column(String(512), comment="Full file storage path (including file name)")
    file_name: Mapped[str] = mapped_column(String(256), comment="Original file name (without extension)")
    file_ext: Mapped[str] = mapped_column(String(256), comment="File extension")
    status: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="File status enumeration type")
    file_version: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="File version number (increments on each update)")
    is_overwrite: Mapped[int] = mapped_column(Numeric(1, 0), comment="Whether to overwrite: 1 - Yes, 0 - No")
    security_level: Mapped[int] = mapped_column(Numeric(1, 0), comment="File security level enumeration type")
    file_size: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="File size (in bytes)")
    chunk_parser: Mapped[dict | None] = mapped_column(OracleJSON, comment="Data chunk parsing parameters")
    biz_metadata: Mapped[dict | None] = mapped_column(OracleJSON, comment="Business metadata in JSON format (stored as string)")
    process_priority: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Processing priority enumeration type")
    log_msg: Mapped[str | None] = mapped_column(String(4000), comment="Processing log information")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")
    approved_by: Mapped[str | None] = mapped_column(String(256), comment="Approver user")
    approved_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Approval time")
    approve_comments: Mapped[str | None] = mapped_column(String(1024), comment="Approval comments")