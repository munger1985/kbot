from typing import Optional

from sqlalchemy import BigInteger, CLOB, String, JSON, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database.oracle import Base

class KBotKBChunks(Base):
    """KBOT_KB_CHUNKS table model."""
    
    __tablename__ = "KBOT_KB_CHUNKS"
    
    id: Mapped[str] = mapped_column(String(256), primary_key=True)
    app_id: Mapped[int] = mapped_column(BigInteger)
    kb_id: Mapped[int] = mapped_column(BigInteger)
    batch_id: Mapped[int] = mapped_column(BigInteger)
    file_id: Mapped[int] = mapped_column(BigInteger)
    chunk_doc: Mapped[str] = mapped_column(CLOB)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # VECTOR类型
    chunk_metadata: Mapped[dict] = mapped_column(JSON)
    chunk_category: Mapped[Optional[str]] = mapped_column(String(256))
    source_chunk_id: Mapped[Optional[str]] = mapped_column(String(256))