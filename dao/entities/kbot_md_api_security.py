from datetime import datetime, timezone
from sqlalchemy import String, Date, Numeric
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base



class KbotMdApiSecurity(Base):
    """API密钥表"""
    
    security_id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="主键id")
    app_id: Mapped[int | None] = mapped_column(Numeric(38, 0), comment="所属app_id")
    accessor: Mapped[str] = mapped_column(String(256), comment="访问者")
    accessor_type: Mapped[int] = mapped_column(Numeric(1, 0), comment="访问者类型, 枚举")
    hashed_secret: Mapped[str] = mapped_column(String(256), comment="加密后的密钥")
    status: Mapped[int | None] = mapped_column(Numeric(2, 0), comment="状态：枚举")
    descs: Mapped[str | None] = mapped_column(String(512), comment="描述")
    created_by: Mapped[str | None] = mapped_column(String(256), comment="创建用户")
    created_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), comment="创建时间")
    updated_by: Mapped[str | None] = mapped_column(String(256), comment="修改用户")
    updated_time: Mapped[Date] = mapped_column(Date, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), comment="修改时间")

    def __repr__(self):
        return f"KbotMdApiSecurity(accessor={self.accessor!r},hashed_secret={self.hashed_secret!r},status={self.status!r})"