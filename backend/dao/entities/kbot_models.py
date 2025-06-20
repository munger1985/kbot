from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotModels(Base):
    """KBOT_KB_MODELS table model."""
    
    __tablename__ = "KBOT_KB_MODELS"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    display_name: Mapped[str] = mapped_column(String(256))
    model_name: Mapped[str] = mapped_column(String(256))
    category: Mapped[str] = mapped_column(String(256))
    provider: Mapped[str] = mapped_column(String(256))
    api_endpoint: Mapped[str] = mapped_column(String(256))
    api_key: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(1))
    model_params: Mapped[dict] = mapped_column(JSON)
    model_template: Mapped[dict] = mapped_column(JSON)
    descs: Mapped[Optional[str]] = mapped_column(String(512))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)