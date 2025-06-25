from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotPrompt(Base):
    """KBOT_PROMPT table model."""
    
    __tablename__ = "KBOT_PROMPT"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    name: Mapped[str] = mapped_column(String(256))
    domain_name: Mapped[str] = mapped_column(String(256))
    prompt_category: Mapped[str] = mapped_column(String(256))
    template: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(1))
    descs: Mapped[Optional[str]] = mapped_column(String(512))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)