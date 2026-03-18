from sqlalchemy import String, DateTime, Numeric, func, Enum, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from core.dictionary import UserTokenStatus
from .base import BaseEntity



class UserToken(BaseEntity):
    """JWT token management entity for user authentication.
    
    Maps to database table `KBOT_SYS_AUTH_TOKEN` and stores user authentication token metadata
    including JWT identifiers, user association, device/IP information, and lifecycle status.
    Used to track, validate, and revoke user tokens for security control.
    """

    __tablename__ = "KBOT_SYS_AUTH_TOKEN"
    
    # Core identification fields
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Auto-increment user token ID (38-digit numeric)")
    jti: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, comment="Unique JWT ID (UUID format, 36 chars)")
    user_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated user ID (FK to user table)")
    
    # Client context fields
    device_info: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="Client device information (200 chars max)")
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True, comment="Client IP address (supports IPv6, 45 chars max)")
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True, comment="Client user agent string (500 chars max)")
    
    # Token lifecycle & status
    status: Mapped[UserTokenStatus] = mapped_column(Enum(UserTokenStatus), default=UserTokenStatus.ACTIVE, comment="Token status (ACTIVE/REVOKED/EXPIRED)")
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, comment="Token expiration timestamp (mandatory, UTC)")
    revoked_reason: Mapped[str | None] = mapped_column(String(200), nullable=True, comment="Reason for token revocation (200 chars max)")
    
    # Time tracking fields
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="Token creation timestamp (auto-generated)")
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="Token revocation timestamp (nullable)")

# Composite index for frequent query patterns (user_id + status + expires_at)
Index("idx_user_token_user_status", UserToken.user_id, UserToken.status, UserToken.expires_at)