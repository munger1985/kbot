from sqlalchemy import String, DateTime, Numeric, func, Enum, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from .base import BaseEntity

class User(BaseEntity):
    """User authentication entity for system access control.
    
    Maps to database table `KBOT_SYS_AUTH` and stores core user authentication data
    including credentials, status flags, and login timestamps.
    Enforces unique constraints on username/email and uses secure password hashing.
    """

    __tablename__ = "KBOT_SYS_AUTH"
    
    # Core identification fields
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, index=True, comment="Auto-increment user ID (38-digit numeric)")
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False, comment="Unique system username (50 chars max)")
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, comment="Unique user email address (100 chars max)")
    
    # Security & authentication
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False, comment="BCrypt-hashed password (never store plaintext)")
    
    # Access control flags (FIX: Numeric → Boolean for semantic clarity)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="User account status (True = active, False = disabled)")
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, comment="Superuser privilege flag (True = full system access)")
    
    # Time tracking fields (FIX: Date → DateTime for precise timestamp)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, comment="Last successful login timestamp (UTC)")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), comment="Account creation timestamp (auto-generated)")
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, onupdate=func.now(), comment="Last account modification timestamp")

# Composite index for frequent login/query patterns
Index("idx_user_login", User.username, User.email, User.is_active)