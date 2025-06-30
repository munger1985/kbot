from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class KbotMdKbBatch(Base):
    """Knowledge base batch entity for KBOT_MD_KB_BATCH table."""

    __table_args__ = (
        UniqueConstraint('batch_name', 'kb_id'),
    )
    
    batch_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="批次唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    batch_name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="批次名称，与KB_ID组成联合唯一约束"
    )
    kb_id: Mapped[int] = mapped_column(
        NUMBER(38, 0),
        ForeignKey("KBOT_MD_KB.kb_id"),
        nullable=False,
        comment="关联的知识库ID，外键引用KBOT_MD_KB表"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="批次创建人"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="批次创建时间，默认系统当前时间"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="最后修改人"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="最后修改时间，默认系统当前时间"
    )