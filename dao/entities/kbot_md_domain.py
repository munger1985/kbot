from datetime import datetime, timezone
from sqlalchemy import String, Date, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base


class KbotMdDomain(Base):
    """业务域表"""
    
    domain_id: Mapped[int] = mapped_column(Numeric(38, 0),  primary_key=True, comment="业务域ID，主键")
    app_id: Mapped[int] = mapped_column(Numeric(38, 0),  nullable=False, comment="所属APP ID，与NAME组成唯一索引")
    name: Mapped[str | None] = mapped_column(String(256), comment="业务域名称")
    status: Mapped[int | None] = mapped_column(Numeric(1, 0), comment="提示词状态：1-启用，0-禁用")
    descs: Mapped[str | None] = mapped_column(String(512), comment="业务域描述")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment="修改时间")

    def __repr__(self):
        return f"KbotMdDomain(domain_id={self.domain_id!r}, app_id={self.app_id!r}, name={self.name!r})"