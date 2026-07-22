import json
import uuid
from typing import Any
from datetime import datetime
from sqlalchemy import String, Numeric, CLOB, DateTime, func, Index
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
    embedding: Mapped[list[float]] = mapped_column(VectorField(), nullable=True, comment="Vector embedding of text chunk")
    chunk_metadata: Mapped[dict] = mapped_column(OracleJSON, nullable=False, comment="Chunk metadata (JSON, e.g. page_num/image_name)")

    # Extended fields
    # Note: OracleJSON is a TypeDecorator that ensures proper JSON serialization
    security_level: Mapped[int | None] = mapped_column(Numeric(1, 0), nullable=True, comment="Data security level (1=public, 0=private, nullable)")
    is_active: Mapped[int] = mapped_column(Numeric(1), default=1, nullable=False, comment="Embedding status (1=active, 0=inactive, default=1)")
    biz_metadata: Mapped[dict | None] = mapped_column(OracleJSON, nullable=True, comment="Business custom metadata (JSON, nullable)")

    # Hierarchy / structure fields
    hierarchy_path: Mapped[list[str] | None] = mapped_column(OracleJSON, nullable=True, comment="Hierarchy path from root to this chunk (JSON array of string IDs)")
    hierarchy_depth: Mapped[int] = mapped_column(Numeric(38, 0), default=0, nullable=True, comment="Depth in document hierarchy")
    heading_level: Mapped[int] = mapped_column(Numeric(38, 0), default=0, nullable=True, comment="Heading/header level in document")
    parent_chunk_id: Mapped[str | None] = mapped_column(String(256), nullable=True, comment="Parent chunk ID for hierarchical structure")
    section_id: Mapped[str | None] = mapped_column(String(255), nullable=True, comment="Section identifier")

    # Timestamp fields
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=True, comment="Record creation timestamp")

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        """Serialize entity to dictionary representation."""
        data = {
            "chunk_id": self.chunk_id,
            "chunk_num": self.chunk_num,
            "chunk_type": self.chunk_type,
            "kb_id": self.kb_id,
            "file_id": self.file_id,
            "content": self.content,
            "header": self.header,
            "doc_summary": self.doc_summary,
            "search_helper": self.search_helper,
            "chunk_metadata": self.chunk_metadata,
            "biz_metadata": self.biz_metadata,
            "security_level": self.security_level,
            "is_active": self.is_active,
            "hierarchy_path": self.hierarchy_path,
            "hierarchy_depth": self.hierarchy_depth,
            "heading_level": self.heading_level,
            "parent_chunk_id": self.parent_chunk_id,
            "section_id": self.section_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

# Performance optimization indexes for vector search
Index("idx_embedding_kb_status", TxtChunkEntity.kb_id, TxtChunkEntity.is_active, TxtChunkEntity.security_level)
Index("idx_embedding_file_id", TxtChunkEntity.file_id)
