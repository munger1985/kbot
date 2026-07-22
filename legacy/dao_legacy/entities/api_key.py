from sqlalchemy import String, DateTime, Numeric, func, Enum, Index, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .base import BaseEntity
from platform_core.dictionary import APIKeyStatus
# 延迟导入，避免循环引用
# from .service import Service


class APIKey(BaseEntity):
    """API Key management entity for service authentication.
    
    Maps to database table `KBOT_SYS_API_KEYS` and stores API key configurations
    including identification, authentication, access control, and usage metrics.
    Maintains foreign key relationship with `KBOT_SYS_SERVICES` (Service entity).
    """

    __tablename__ = "KBOT_SYS_API_KEYS"
    
    # Core identification fields
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Auto-increment primary key (38-digit numeric)")
    key_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False, comment="Public unique API key identifier (50 chars max)")
    hashed_key: Mapped[str] = mapped_column(String(255), nullable=False, comment="BCrypt-hashed secret key (never store plaintext)")
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False, comment="Original key prefix (first 8 chars for display)")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="Human-readable API key name/description (100 chars max)")
    
    # Foreign key & access control
    service_id: Mapped[int] = mapped_column(Numeric(38, 0), ForeignKey("KBOT_SYS_SERVICES.id"), nullable=False, index=True, comment="Associated service ID (FK to KBOT_SYS_SERVICES.id)")
    scopes: Mapped[str] = mapped_column(String(4000), default="[]", comment="API access scopes (JSON array of permitted endpoints)")
    status: Mapped[APIKeyStatus] = mapped_column(Enum(APIKeyStatus), default=APIKeyStatus.ACTIVE, index=True, comment="API key status (ACTIVE/REVOKED/EXPIRED)")
    
    # Time-related fields (FIX: Date → DateTime for precise timestamp)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True, comment="Expiration timestamp (None = never expires)")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="Last API key usage timestamp")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="Record creation timestamp (auto-generated)")
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, onupdate=func.now(), comment="Last update timestamp (auto-updated)")
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="API key revocation timestamp")
    
    # Usage & restriction fields
    usage_count: Mapped[int] = mapped_column(Numeric(38, 0), default=0, comment="Total API key usage count (auto-increment)")
    allowed_ips: Mapped[str] = mapped_column(String(4000), default="[]", comment="IP whitelist (JSON array, empty = no restriction)")
    rate_limit: Mapped[int] = mapped_column(Numeric(38, 0), default=0, comment="Rate limit (requests/minute, 0 = unlimited)")
    
    # Audit fields
    created_by: Mapped[str | None] = mapped_column(String(50), nullable=True, comment="Creator username (50 chars max)")
    revoked_reason: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="Reason for API key revocation (500 chars max)")
    
    # Relationship definition (bidirectional with Service entity)
    # 使用字符串形式的类型引用，避免循环导入
    service: Mapped["Service"] = relationship("Service", back_populates="api_keys") # type: ignore

# Composite index for frequent query patterns (optional but recommended)
Index("idx_api_key_service_status", APIKey.service_id, APIKey.status, APIKey.expires_at)