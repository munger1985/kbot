from datetime import datetime, timezone
from sqlalchemy import String, Date, Numeric, CLOB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdParserConf(Base):
    """System configuration entity for KBOT_MD_PARSER_CONF table."""
    
    conf_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="配置项唯一标识，主键")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属应用ID")
    kb_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属知识库ID")
    file_category: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="文件类别：1-text；2-image；3-audio；4-video")
    file_ext: Mapped[str | None] = mapped_column(String(100), comment="文件扩展名（小写）。如txt, jpg, png, mp3, mp4")
    chunk_parser: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="Chunk拆分策略的编码列表，以逗号隔开")
    chunk_parser_param: Mapped[str | None] = mapped_column(CLOB, comment="Chunk参数")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment="修改时间")
