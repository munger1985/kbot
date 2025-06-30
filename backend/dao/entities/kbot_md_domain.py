from sqlalchemy import String, Date
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.sql import func
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class KbotMdDomain(Base):
    """Business domain entity for KBOT_MD_DOMAIN table."""
    __table_args__ = (
        UniqueConstraint('app_id', 'name'),
    )
    
    domain_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        primary_key=True,
        comment="业务域ID，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属APP ID，与NAME组成唯一索引"
    )
    name: Mapped[str | None] = mapped_column(
        String(256),
        comment="业务域名称"
    )
    status: Mapped[str | None] = mapped_column(
        String(1),
        comment="提示词状态：Y-启用，N-禁用"
    )
    descs: Mapped[str | None] = mapped_column(
        String(512),
        comment="业务域描述"
    )
    created_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="创建用户"
    )
    created_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="创建时间，默认当前日期"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(512),
        comment="修改用户"
    )
    updated_time: Mapped[Date] = mapped_column(
        Date, 
        server_default=func.current_date(),
        comment="修改时间，默认当前日期"
    )