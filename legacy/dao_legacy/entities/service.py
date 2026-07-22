from sqlalchemy import String, DateTime, Numeric, func, Enum, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .base import BaseEntity
from platform_core.dictionary import ServiceType
# 延迟导入，避免循环引用
# from .api_key import APIKey

class Service(BaseEntity):
    """Service management entity for API access control.
    
    Maps to database table `KBOT_SYS_SERVICES` and stores core service metadata
    including identification, classification, ownership, and status.
    Establishes one-to-many relationship with `APIKey` entity (one service → multiple API keys).
    """

    __tablename__ = "KBOT_SYS_SERVICES"
    
    # Core identification fields
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True, comment="Auto-increment primary key (38-digit numeric)")
    service_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="Unique service identifier (e.g., 'order-service', 50 chars max)")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="Human-readable service name (100 chars max)")
    description: Mapped[str | None] = mapped_column(String(4000), nullable=True, comment="Detailed service description (4000 chars max, nullable)")
    
    # Classification & ownership
    service_type: Mapped[ServiceType] = mapped_column(Enum(ServiceType, name="service_type_enum"), default=ServiceType.INTERNAL, comment="Service type (INTERNAL/EXTERNAL/THIRD_PARTY)")
    owner: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="Service owner username (100 chars max, nullable)")
    contact_email: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="Owner contact email (100 chars max, nullable)")
    
    # Status & time fields (FIX: Date → DateTime for precise timestamp; Numeric → Boolean for readability)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, comment="Service status flag (True = active, False = inactive)")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.current_timestamp(), comment="Record creation timestamp (auto-generated)")
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, onupdate=func.current_timestamp(), nullable=True, comment="Last update timestamp (auto-updated)")
    
    # Relationship definition (one-to-many with APIKey, explicit type hint)
    # 使用字符串形式的类型引用，避免循环导入
    api_keys: Mapped[list["APIKey"]] = relationship("APIKey", back_populates="service", cascade="all, delete-orphan") # type: ignore