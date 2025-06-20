from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotKBBatch(Base):
    """KBOT_KB_BATCH table model."""
    
    __tablename__ = "KBOT_KB_BATCH"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    batch_name: Mapped[str] = mapped_column(String(256))
    kb_id: Mapped[int] = mapped_column(BigInteger)
    parse_engine: Mapped[str] = mapped_column(String(256))
    engine_params: Mapped[dict] = mapped_column(JSON)
    enable_summary: Mapped[int] = mapped_column(BigInteger)
    metadata: Mapped[dict] = mapped_column(JSON)
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)