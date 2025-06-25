from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotKB(Base):
    """KBOT_KB table model."""
    
    __tablename__ = "KBOT_KB"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    domain_id: Mapped[int] = mapped_column(BigInteger)
    kb_name: Mapped[str] = mapped_column(String(256))
    kb_category: Mapped[str] = mapped_column(String(256))
    descs: Mapped[Optional[str]] = mapped_column(String(512))
    vecdb_conn_id: Mapped[int] = mapped_column(BigInteger)
    embed_model_id: Mapped[int] = mapped_column(BigInteger)
    kb_status: Mapped[str] = mapped_column(String(256))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)