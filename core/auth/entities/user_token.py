from sqlalchemy import String, Date, Numeric, func, Index
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base
from sqlalchemy.sql import func
from datetime import datetime
from core.dictionary import UserTokenStatus
from sqlalchemy import Enum


class UserToken(Base):

    __tablename__ = "KBOT_MD_USER_TOKEN"
    
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="用户令牌ID")
    jti: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, comment="JWT ID")
    user_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="所属用户")
    device_info: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="设备信息")
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True, comment="IP地址")
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="用户代理")
    status: Mapped[UserTokenStatus] = mapped_column(Enum(UserTokenStatus), default=UserTokenStatus.ACTIVE, index=True, comment="令牌状态")
    expires_at: Mapped[datetime] = mapped_column(Date, nullable=False, index=True, comment="过期时间")
    revoked_reason: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="撤销原因")
    created_at: Mapped[datetime] = mapped_column(Date, server_default=func.now(), comment="创建时间")
    revoked_at: Mapped[datetime | None] = mapped_column(Date, nullable=True, comment="撤销时间")
