from sqlalchemy import String, Date, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .base import Base

class User(Base):
    __tablename__ = "KBOT_SYS_AUTH"
    
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, index=True, comment="用户ID")
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False, comment="用户名")
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="邮箱")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码哈希")
    is_active: Mapped[bool] = mapped_column(Numeric(1,0), default=True, comment="是否激活")
    is_superuser: Mapped[bool] = mapped_column(Numeric(1,0), default=False, comment="是否超级用户")
    last_login_at: Mapped[datetime | None] = mapped_column(Date, nullable=True, comment="最后登录时间")
    created_at: Mapped[datetime] = mapped_column(Date, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime | None] = mapped_column(Date, onupdate=func.now(), comment="更新时间")