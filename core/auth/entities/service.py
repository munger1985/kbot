from sqlalchemy import String, Date, Numeric, func, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from .base import Base
from core.dictionary import ServiceType

class Service(Base):
    __tablename__ = "KBOT_MD_SERVICES"
    
    id: Mapped[int] = mapped_column(Numeric(38, 0), primary_key=True)
    service_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="服务代码，如: 'order-service', 'payment-service'")
    name: Mapped[str] = mapped_column(String(100), nullable=False, comment="服务名称")
    description: Mapped[str | None] = mapped_column(String(4000), nullable=True, comment="服务描述")
    service_type: Mapped[ServiceType] = mapped_column(Enum(ServiceType, name="service_type_enum"), default=ServiceType.INTERNAL, comment="服务类型")
    owner: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="负责人")
    contact_email: Mapped[str | None] = mapped_column(String(100), nullable=True, comment="联系邮箱")
    is_active: Mapped[bool] = mapped_column(Numeric(1,0), default=True, comment="是否激活")
    created_at: Mapped[datetime] = mapped_column(Date, server_default=func.current_timestamp(), comment="创建时间")
    updated_at: Mapped[datetime | None] = mapped_column(Date, onupdate=func.current_timestamp(), nullable=True, comment="更新时间")
    
    # 关系定义
    api_keys: Mapped[list] = relationship("APIKey", back_populates="service")
    