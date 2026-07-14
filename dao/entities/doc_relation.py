"""
Oracle 23ai 兼容的文档引用关系实体

映射表 doc_relation，记录文档间的引用/依赖/替代关系。
"""
from sqlalchemy import String, Numeric, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON


class DocRelationEntity(BaseEntity):
    """文档引用关系表 (doc_relation) 实体"""

    __tablename__ = "doc_relation"

    id: Mapped[int] = mapped_column(
        Numeric(38, 0), primary_key=True, autoincrement=True,
        comment="主键"
    )
    kb_id: Mapped[int] = mapped_column(
        Numeric(38, 0), nullable=False,
        comment="所属知识库 ID"
    )
    source_file_id: Mapped[str] = mapped_column(
        String(256), nullable=False,
        comment="引用方文件 ID"
    )
    target_file_id: Mapped[str | None] = mapped_column(
        String(256),
        comment="被引用方文件 ID（同知识库内已知文件时填充）"
    )
    target_doc_name: Mapped[str | None] = mapped_column(
        String(1000),
        comment="被引用文档名称"
    )
    target_chapter: Mapped[str | None] = mapped_column(
        String(500),
        comment="被引用的章节号，如 第3章"
    )
    target_section: Mapped[str | None] = mapped_column(
        String(500),
        comment="被引用的节号，如 3.1.2"
    )
    relation_type: Mapped[str] = mapped_column(
        String(64), default="reference",
        comment="关系类型：reference/dependency/replace"
    )
    context_snippet: Mapped[str | None] = mapped_column(
        Text,  # Oracle CLOB 由 Text 自动映射
        comment="引用处的上下文原文（1-2 句）"
    )
    confidence: Mapped[float | None] = mapped_column(
        Numeric(3, 2), default=1.0,
        comment="LLM 提取的置信度，0~1"
    )
    biz_metadata: Mapped[dict | None] = mapped_column(
        OracleJSON, default=dict,
        comment="业务扩展字段"
    )
    created_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        comment="创建时间"
    )
