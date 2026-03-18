from sqlalchemy import String, Date, Numeric, CLOB, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity

class SysParserConfEntity(BaseEntity):
    """Table for System-level Parser Configuration.

    This entity maps to the database table `kbot_sys_parser_conf` and stores global system-level
    file parsing configurations, including file category mapping, extension rules, chunk splitting
    strategies, default configuration flags, and audit timestamps.
    """

    __tablename__ = "kbot_sys_parser_conf"
    
    conf_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Unique configuration item identifier, primary key")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated application ID")
    file_category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="File category: 1 - text; 2 - image; 3 - audio; 4 - video")
    file_ext: Mapped[str | None] = mapped_column(String(100), comment="File extension (lowercase). e.g., txt, jpg, png, mp3, mp4")
    chunk_parser: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Coded list of chunk splitting strategies (separated by commas)")
    chunk_parser_param: Mapped[str | None] = mapped_column(CLOB, comment="Chunk parsing parameters")
    is_default: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Whether it is default configuration: 1 - Yes, 0 - No")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="Configuration status: 1 - Enabled, 0 - Disabled")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="Creator user")
    created_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), comment="Creation time")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="Updater user")
    updated_time: Mapped[Date] = mapped_column(Date, server_default=func.now(), onupdate=func.now(), comment="Update time")