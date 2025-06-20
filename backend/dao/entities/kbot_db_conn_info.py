from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotDbConnInfo(Base):
    """KBOT_DB_CONN_INFO table model."""
    
    __tablename__ = "KBOT_DB_CONN_INFO"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    db_display_name: Mapped[str] = mapped_column(String(256))
    db_type: Mapped[str] = mapped_column(String(20))
    db_conn_str: Mapped[dict] = mapped_column(JSON)
    descs: Mapped[Optional[str]] = mapped_column(String(512))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)