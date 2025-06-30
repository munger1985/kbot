from enum import Enum
from sqlalchemy import String
from sqlalchemy.dialects.oracle import NUMBER
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base

class ChunkType(int, Enum):
    """Chunk type enumeration."""
    TEXT = 1
    IMAGE = 2

class KbotMdKbChunks(Base):
    """Knowledge base chunks entity for KBOT_MD_KB_CHUNKS table."""
    
    chunk_id: Mapped[str] = mapped_column(
        String(256), 
        primary_key=True,
        comment="分块唯一标识，主键"
    )
    app_id: Mapped[int] = mapped_column(
        NUMBER(38, 0), 
        nullable=False,
        comment="所属应用ID"
    )
    kb_id: Mapped[int] = mapped_column(
        NUMBER(38, 0),
        nullable=False,
        comment="关联的知识库ID"
    )
    batch_id: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="关联的批次ID（可选）"
    )
    file_id: Mapped[int] = mapped_column(
        NUMBER(38, 0),
        nullable=False,
        comment="关联的原始文件ID"
    )
    chunk_type: Mapped[int | None] = mapped_column(
        NUMBER(38, 0),
        comment="分块类型：1-文本(TXT)、2-图片(IMG)"
    )