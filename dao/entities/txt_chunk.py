import json
import uuid
from typing import Any
from sqlalchemy import String, Numeric, CLOB, Index
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, VectorField, OracleJSON


class TxtChunkEntity(BaseEntity):
    """Text chunk embedding storage entity for knowledge base.
    Maps to database table `KBOT_BIZ_TXT_EMBEDDING` and stores text chunk data,
    corresponding vector embeddings, and metadata for knowledge base retrieval.
    Supports vector similarity search and multi-level security control.
    """
    __tablename__ = "KBOT_BIZ_TXT_EMBEDDING"
    
    # Core identification fields
    chunk_id: Mapped[str] = mapped_column(String(256), primary_key=True, comment="Unique embedding ID (primary key, 256 chars max)")
    kb_id: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Associated knowledge base ID")
    file_id: Mapped[str] = mapped_column(String(256), nullable=False, comment="Source file ID (256 chars max)")
    chunk_num: Mapped[int] = mapped_column(Numeric(38, 0), nullable=False, comment="Chunk number (sequence)")
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=True, comment="Chunk type (text, table, picture, slide)")

    # Core business fields (vector/text/metadata)
    content: Mapped[str] = mapped_column(CLOB, nullable=False, comment="Original text chunk content (CLOB, mandatory)")
    header: Mapped[str] = mapped_column(String(256), nullable=True, comment="Header of chunk content")
    doc_summary: Mapped[str] = mapped_column(String(4000), nullable=True, comment="Summary of document content")
    search_helper: Mapped[str] = mapped_column(String(4000), nullable=True, comment="Search helper of chunk content")
    embedding: Mapped[list[float]] = mapped_column(VectorField(), nullable=False, comment="Vector embedding of text chunk")
    chunk_metadata: Mapped[dict] = mapped_column(OracleJSON, nullable=False, comment="Chunk metadata (JSON, e.g. page_num/image_name)")

    # Extended fields
    # Note: OracleJSON is a TypeDecorator that ensures proper JSON serialization
    security_level: Mapped[int | None] = mapped_column(Numeric(1, 0), nullable=True, comment="Data security level (1=public, 0=private, nullable)")
    is_active: Mapped[int] = mapped_column(Numeric(1), default=1, nullable=False, comment="Embedding status (1=active, 0=inactive, default=1)")
    biz_metadata: Mapped[dict | None] = mapped_column(OracleJSON, nullable=True, comment="Business custom metadata (JSON, nullable)")

# Performance optimization indexes for vector search
Index("idx_embedding_kb_status", TxtChunkEntity.kb_id, TxtChunkEntity.is_active, TxtChunkEntity.security_level)
Index("idx_embedding_file_id", TxtChunkEntity.file_id)