from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotKBFiles(Base):
    """KBOT_KB_FILES table model."""
    
    __tablename__ = "KBOT_KB_FILES"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    kb_id: Mapped[int] = mapped_column(BigInteger)
    batch_id: Mapped[int] = mapped_column(BigInteger)
    file_path: Mapped[str] = mapped_column(String(512))
    file_name: Mapped[str] = mapped_column(String(256))
    file_ext: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(100))
    file_version: Mapped[int] = mapped_column(BigInteger)
    is_overwrite: Mapped[str] = mapped_column(String(1))
    security_level: Mapped[Optional[str]] = mapped_column(String(256))
    file_size: Mapped[int] = mapped_column(BigInteger)
    chunks_cnt: Mapped[int] = mapped_column(BigInteger)
    log_msg: Mapped[Optional[str]] = mapped_column(String(1000))
    created_by: Mapped[str] = mapped_column(String(512))
    created_time: Mapped[datetime] = mapped_column(DateTime)
    updated_by: Mapped[str] = mapped_column(String(512))
    updated_time: Mapped[datetime] = mapped_column(DateTime)
    approved_by: Mapped[Optional[str]] = mapped_column(String(256))
    approved_time: Mapped[Optional[datetime]] = mapped_column(DateTime)