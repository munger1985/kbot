import json
import uuid
from typing import Any
from sqlalchemy import String, Numeric, CLOB, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, VectorField


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

    # Core business fields (vector/text/metadata)
    content: Mapped[str] = mapped_column(CLOB, nullable=False, comment="Original text chunk content (CLOB, mandatory)")
    structure_level: Mapped[int] = mapped_column(Numeric(38, 0), nullable=True, comment="Structure level (document depth)")
    path_names: Mapped[str] = mapped_column(String(4000), nullable=True, comment="Path names (e.g., chapter/section hierarchy)")
    chunk_type: Mapped[str] = mapped_column(String(20), nullable=True, comment="Chunk type (text, table, picture, heading)")
    embedding: Mapped[list[float]] = mapped_column(VectorField(), nullable=False, comment="Vector embedding of text chunk")
    chunk_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, comment="Chunk metadata (JSON, e.g. position/length/source)")
    security_level: Mapped[int | None] = mapped_column(Numeric(1, 0), nullable=True, comment="Data security level (1=public, 0=private, nullable)")
    is_active: Mapped[int] = mapped_column(Numeric(1), default=1, nullable=False, comment="Embedding status (1=active, 0=inactive, default=1)")

    # Extended fields
    biz_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True, comment="Business custom metadata (JSON, nullable)")
# Performance optimization indexes for vector search
Index("idx_embedding_kb_status", TxtChunkEntity.kb_id, TxtChunkEntity.is_active, TxtChunkEntity.security_level)
Index("idx_embedding_file_id", TxtChunkEntity.file_id)


class TxtChunk:
    """
    文本向量嵌入实体类 - 适配分层检索架构
    对接 Docling 解析输出与 Elasticsearch 9.x 存储
    """
    def __init__(self, 
                 chunk_id: str, 
                 kb_id: str, 
                 file_id: str, 
                 content: str, 
                 structure_level: int,      # 对应文档树深度 (L1, L2...)
                 path_names: list[str],     # 对应 OpenViking 路径基因 (e.g. ["第一章", "1.1 节"])
                 embedding: list[float],    # 向量数据
                 chunk_metadata: dict[str, Any], 
                 biz_metadata: dict[str, Any],
                 security_level: int,
                 chunk_type: str = "text",  # 类型：text, table, picture, heading
                 is_active: bool = True):
        
        self.chunk_id = chunk_id
        self.kb_id = kb_id
        self.file_id = file_id
        self.content = content
        self.structure_level = structure_level
        self.path_names = path_names
        self.embedding = embedding
        self.chunk_metadata = chunk_metadata
        self.biz_metadata = biz_metadata
        self.security_level = security_level
        self.chunk_type = chunk_type
        self.is_active = is_active
         
    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        """转换为 ES 存储字典"""
        data = {
            "chunk_id": self.chunk_id,
            "kb_id": self.kb_id,
            "file_id": self.file_id,
            "content": self.content,
            "structure_level": self.structure_level,
            "path_names": self.path_names,  # ES 索引中作为 Keyword 数组
            "chunk_type": self.chunk_type,
            "chunk_metadata": self.chunk_metadata,
            "biz_metadata": self.biz_metadata,
            "security_level": self.security_level,
            "is_active": self.is_active
        }
        if include_embedding:
            data["embedding"] = self.embedding
        return data
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_docling_chunk(cls, docling_chunk: dict[str, Any], **kwargs) -> 'TxtChunk':
        """
        工厂方法：直接将 Docling 处理器输出映射到实体类
        kwargs 通常包含 kb_id, file_id, embedding, security_level 等业务信息
        """
        return cls(
            chunk_id=str(uuid.uuid4()), # 或使用特定算法生成
            kb_id=kwargs.get("kb_id"),  # type: ignore
            file_id=kwargs.get("file_id"),  # type: ignore
            content=docling_chunk["content"],
            structure_level=docling_chunk["structure_level"],
            path_names=docling_chunk["path_names"],
            embedding=kwargs.get("embedding", []),
            chunk_metadata=docling_chunk["metadata"],
            biz_metadata=kwargs.get("biz_metadata", {}),
            security_level=kwargs.get("security_level", 0),
            chunk_type=docling_chunk["chunk_type"]
        )