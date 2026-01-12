from sqlalchemy import String, Date, Numeric, func, Enum, Index, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .base import Base
from core.dictionary import APIKeyStatus
from .service import Service


class APIKey(Base):

    __tablename__ = "KBOT_SYS_API_KEYS"
    
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    key_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False, comment="密钥ID（公开部分，用于识别）")
    hashed_key: Mapped[str] = mapped_column(String(255), nullable=False, comment="哈希后的密钥（不存储明文）")
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False, comment="原始密钥前缀，用于显示给用户（只存储前8位）")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="密钥名称/描述")
    service_id: Mapped[int] = mapped_column(Numeric(38, 0), ForeignKey("KBOT_SYS_SERVICES.id"), nullable=False, index=True, comment="关联服务ID")
    scopes: Mapped[str] = mapped_column(String(4000), default="[]", comment="权限范围（JSON格式，存储该key能访问的API列表）")
    status: Mapped[APIKeyStatus] = mapped_column(Enum(APIKeyStatus), default=APIKeyStatus.ACTIVE, index=True, comment="密钥状态")
    expires_at: Mapped[datetime | None] = mapped_column(Date, nullable=True, index=True, comment="过期时间（None表示永不过期）")
    last_used_at: Mapped[datetime | None] = mapped_column(Date, nullable=True, comment="最后使用时间")
    usage_count: Mapped[int] = mapped_column(Numeric(38, 0), default=0, comment="使用次数")
    allowed_ips: Mapped[str] = mapped_column(String(4000), default="[]", comment="IP白名单（JSON数组，空表示不限制）")
    rate_limit: Mapped[int] = mapped_column(Numeric(38, 0), default=0, comment="速率限制（每分钟请求数，0表示不限制）")
    created_by: Mapped[int | None] = mapped_column(Numeric(38, 0), nullable=True, comment="创建者ID")
    created_at: Mapped[datetime] = mapped_column(Date, server_default=func.now(), comment="创建时间")
    updated_at: Mapped[datetime | None] = mapped_column(Date, onupdate=func.now(), comment="更新时间")
    revoked_reason: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="撤销原因")
    revoked_at: Mapped[datetime | None] = mapped_column(Date, nullable=True, comment="撤销时间")
    
    # 关系定义
    service: Mapped[Service] = relationship("Service", back_populates="api_keys")
