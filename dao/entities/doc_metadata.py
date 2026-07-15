"""
Oracle 23ai 兼容的文档元数据实体

映射表 doc_metadata，存储从文档中提取的结构化元信息。
"""
from sqlalchemy import String, Date, Numeric, Text, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from .base import BaseEntity, OracleJSON


class DocMetadataEntity(BaseEntity):
    """文档级元数据表 (doc_metadata) 实体"""

    __tablename__ = "kbot_doc_metadata"

    id: Mapped[int] = mapped_column(
        Numeric(38, 0), primary_key=True, autoincrement=True,
        comment="主键"
    )
    kb_id: Mapped[int] = mapped_column(
        Numeric(38, 0), nullable=False,
        comment="所属知识库 ID"
    )
    file_id: Mapped[str] = mapped_column(
        String(256), nullable=False, unique=True,
        comment="关联的文件 ID（唯一约束）"
    )
    doc_name: Mapped[str | None] = mapped_column(
        String(1000),
        comment="文档正式名称（LLM 从标题/文件名提取）"
    )
    doc_type: Mapped[str | None] = mapped_column(
        String(64),
        comment="文档类型：standard/report/contract/manual/other"
    )
    doc_number: Mapped[str | None] = mapped_column(
        String(256),
        comment="标准号/编号，如 Q/XXX-2024-B"
    )
    doc_version: Mapped[str | None] = mapped_column(
        String(64),
        comment="版本号，如 V2.0"
    )
    doc_date: Mapped[Date | None] = mapped_column(
        Date,
        comment="文档日期"
    )
    page_count: Mapped[int | None] = mapped_column(
        Numeric(38, 0),
        comment="文档总页数"
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Numeric(38, 0),
        comment="解析产生的块数"
    )
    doc_abstract: Mapped[str | None] = mapped_column(
        Text,  # Oracle CLOB 由 String/Text 自动映射
        comment="文档摘要"
    )
    doc_keywords: Mapped[list | None] = mapped_column(
        OracleJSON, default=list,
        comment="关键词列表 (JSON 数组)"
    )
    doc_references: Mapped[list | None] = mapped_column(
        OracleJSON, default=list,
        comment="引用列表 (JSON 数组)，含 doc_name/chapter/section/context"
    )
    biz_metadata: Mapped[dict | None] = mapped_column(
        OracleJSON, default=dict,
        comment="业务元数据扩展字段"
    )
    created_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        comment="创建时间"
    )
    updated_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        comment="更新时间"
    )
