from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotSysConf(Base):
    """KBOT_SYS_CONF table model."""
    
    __tablename__ = "KBOT_SYS_CONF"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    param_name: Mapped[str] = mapped_column(String(256))
    param_type: Mapped[str] = mapped_column(String(256))
    param_value: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(1))
    descs: Mapped[Optional[str]] = mapped_column(String(512))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)
    